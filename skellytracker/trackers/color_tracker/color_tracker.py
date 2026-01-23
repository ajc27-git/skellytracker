import logging
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
from pydantic import BaseModel

from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject
from skellytracker.trackers.color_tracker.color_recorder import ColorRecorder

logger = logging.getLogger(__name__)


class ColorMarkerConfig(BaseModel):
    """Configuration for a single color marker."""
    enabled: bool = False
    target_color_bgr: Tuple[int, int, int] = (0, 0, 255)  # Default: Red
    hue_tolerance: int = 20
    saturation_tolerance: int = 70
    value_tolerance: int = 70
    marker_name: str = "marker_1"
    min_contour_area: int = 100


class ColorPatch(BaseModel):
    area: float
    centroid_x: int
    centroid_y: int
    color_match_percentage: float
    marker_index: int  # Which marker this patch belongs to


class ColorTracker(BaseTracker):
    def __init__(
        self,
        marker_configs: Optional[List[ColorMarkerConfig]] = None,
        use_morphological_ops: bool = True,
    ):
        """
        Initialize a multi-color marker tracker.
        
        Args:
            marker_configs: List of configurations for up to 3 markers
            use_morphological_ops: Whether to use morphological operations to clean masks
        """

        # DEBUG: Log what colors we received
        for i, config in enumerate(marker_configs):
            if config.enabled:
                logger.debug(f"DEBUG - Marker {i} config:")
                logger.debug(f"  Name: {config.marker_name}")
                logger.debug(f"  Target BGR from config: {config.target_color_bgr}")
                logger.debug(f"  Enabled: {config.enabled}")
                logger.debug(f"  Hue Tolerance: {config.hue_tolerance}")
                logger.debug(f"  Saturation Tolerance: {config.saturation_tolerance}")
                logger.debug(f"  Value Tolerance: {config.value_tolerance}")
                
                # Convert and show HSV
                target_color_hsv = cv2.cvtColor(
                    np.uint8([[config.target_color_bgr]]), cv2.COLOR_BGR2HSV
                )[0][0]
                logger.debug(f"  Target HSV: {target_color_hsv}")
                
                # Calculate and show bounds
                lower_bound = np.array([
                    max(0, target_color_hsv[0] - config.hue_tolerance),
                    max(0, target_color_hsv[1] - config.saturation_tolerance),
                    max(0, target_color_hsv[2] - config.value_tolerance)
                ])
                upper_bound = np.array([
                    min(179, target_color_hsv[0] + config.hue_tolerance),
                    min(255, target_color_hsv[1] + config.saturation_tolerance),
                    min(255, target_color_hsv[2] + config.value_tolerance)
                ])
                logger.debug(f"  Lower bound: {lower_bound}")
                logger.debug(f"  Upper bound: {upper_bound}")

        # Default configuration for 3 markers if none provided
        if marker_configs is None:
            marker_configs = [
                ColorMarkerConfig(
                    enabled=True,
                    target_color_bgr=(0, 0, 255),  # Red
                    hue_tolerance=20,
                    saturation_tolerance=70,
                    value_tolerance=70,
                    marker_name="red_marker",
                    min_contour_area=100,
                ),
                ColorMarkerConfig(
                    enabled=False,
                    target_color_bgr=(0, 255, 0),  # Green
                    hue_tolerance=20,
                    saturation_tolerance=70,
                    value_tolerance=70,
                    marker_name="green_marker",
                    min_contour_area=100,
                ),
                ColorMarkerConfig(
                    enabled=False,
                    target_color_bgr=(255, 0, 0),  # Blue
                    hue_tolerance=20,
                    saturation_tolerance=70,
                    value_tolerance=70,
                    marker_name="blue_marker",
                    min_contour_area=100,
                ),
            ]
        
        # Ensure we only have up to 3 markers
        if len(marker_configs) > 3:
            logger.warning(f"Only 3 markers supported, using first 3 from {len(marker_configs)}")
            marker_configs = marker_configs[:3]
        
        self.marker_configs = marker_configs
        self.use_morphological_ops = use_morphological_ops
        
        # Get enabled marker names for tracked objects
        enabled_marker_names = [
            config.marker_name for config in marker_configs if config.enabled
        ]
        
        super().__init__(
            tracked_object_names=enabled_marker_names,
            recorder=ColorRecorder(),
        )
        
        # Pre-calculate HSV bounds for each marker
        self.hsv_bounds = []
        for config in marker_configs:
            if config.enabled:
                target_color_hsv = cv2.cvtColor(
                    np.uint8([[config.target_color_bgr]]), cv2.COLOR_BGR2HSV
                )[0][0]
                
                lower_bound = np.array([
                    max(0, target_color_hsv[0] - config.hue_tolerance),
                    max(0, target_color_hsv[1] - config.saturation_tolerance),
                    max(0, target_color_hsv[2] - config.value_tolerance)
                ])
                
                upper_bound = np.array([
                    min(179, target_color_hsv[0] + config.hue_tolerance),
                    min(255, target_color_hsv[1] + config.saturation_tolerance),
                    min(255, target_color_hsv[2] + config.value_tolerance)
                ])
                
                self.hsv_bounds.append({
                    'lower': lower_bound,
                    'upper': upper_bound,
                    'config': config,
                    'target_color_bgr': config.target_color_bgr,
                    'target_color_hsv': target_color_hsv,
                })
        
        logger.info(f"Initialized ColorTracker with {len(enabled_marker_names)} enabled markers")
        for i, bounds in enumerate(self.hsv_bounds):
            logger.info(f"  Marker {i+1}: {bounds['config'].marker_name}")
            logger.info(f"    BGR: {bounds['target_color_bgr']}")
            logger.info(f"    HSV: {bounds['target_color_hsv']}")
            logger.info(f"    Hue Tolerance: {bounds['config'].hue_tolerance}")
            logger.info(f"    Saturation Tolerance: {bounds['config'].saturation_tolerance}")
            logger.info(f"    Value Tolerance: {bounds['config'].value_tolerance}")

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply morphological operations to clean the mask."""
        if not self.use_morphological_ops:
            return mask
            
        kernel = np.ones((5, 5), np.uint8)
        
        # Remove small noise (opening = erosion followed by dilation)
        cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Fill small holes (closing = dilation followed by erosion)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel)
        
        return cleaned_mask

    def _calculate_color_match_percentage(
        self, contour: np.ndarray, mask: np.ndarray
    ) -> float:
        """Calculate what percentage of pixels in the contour match the target color."""
        # Create a mask for just this contour
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)
        
        # Count matching pixels within the contour
        matching_pixels_in_contour = cv2.countNonZero(cv2.bitwise_and(mask, contour_mask))
        total_pixels_in_contour = cv2.countNonZero(contour_mask)
        
        if total_pixels_in_contour == 0:
            return 0.0
            
        return (matching_pixels_in_contour / total_pixels_in_contour) * 100

    def _detect_single_color(
        self, 
        hsv_image: np.ndarray, 
        bounds: Dict,
        marker_index: int
    ) -> Optional[ColorPatch]:
        """
        Detect the most prominent object of a specific color.
        
        Returns:
            ColorPatch for the largest detected object of this color, or None
        """
        lower = bounds['lower']
        upper = bounds['upper']
        target_h = bounds['target_color_hsv'][0]
        tol_h = bounds['config'].hue_tolerance

        # Check for Red wrap-around (Hue near 0 or 180)
        if target_h < tol_h or target_h > (180 - tol_h):
            mask1 = cv2.inRange(hsv_image, lower, upper)
            
            # Calculate the wrap-around bounds
            lower2, upper2 = lower.copy(), upper.copy()
            if target_h < tol_h: # Low end: catch the 170-180 range
                lower2[0], upper2[0] = 180 - (tol_h - target_h), 180
            else: # High end: catch the 0-10 range
                lower2[0], upper2[0] = 0, target_h + tol_h - 180
                
            mask2 = cv2.inRange(hsv_image, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv_image, lower, upper)
        
        # Clean the mask
        if self.use_morphological_ops:
            mask = self._clean_mask(mask)
        
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Filter contours
        valid_candidates = []
        MIN_CIRCULARITY = 0.1  # Hardcoded threshold
        MIN_SOLIDITY = 0.25  # Hardcoded threshold

        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < bounds['config'].min_contour_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Circularity filter
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            # Solidity filter
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 0

            if circularity >= MIN_CIRCULARITY and solidity >= MIN_SOLIDITY:
                valid_candidates.append((contour, area))

        if not valid_candidates:
            return None
        
        # Find the largest contour
        largest_contour, area = max(valid_candidates, key=lambda x: x[1])
        
        # Calculate centroid
        moments = cv2.moments(largest_contour)
        if moments["m00"] == 0:
            return None
            
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
        
        # Calculate color match percentage
        color_match_percentage = self._calculate_color_match_percentage(largest_contour, mask)
        
        return ColorPatch(
            area=area,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            color_match_percentage=color_match_percentage,
            marker_index=marker_index,
        )

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        """
        Process an image to detect multiple colored markers.
        
        Args:
            image: Input BGR image
            
        Returns:
            Dictionary of tracked objects
        """
        # Convert BGR to HSV once for all markers
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Reset all tracked objects to None first
        for marker_name in self.tracked_objects:
            self.tracked_objects[marker_name].pixel_x = None
            self.tracked_objects[marker_name].pixel_y = None
            self.tracked_objects[marker_name].extra["area"] = 0
            self.tracked_objects[marker_name].extra["color_match_percentage"] = 0
            self.tracked_objects[marker_name].extra["detected"] = False
        
        # Detect each enabled marker
        for i, bounds in enumerate(self.hsv_bounds):
            marker_name = bounds['config'].marker_name
            
            # Detect the most prominent object of this color
            patch = self._detect_single_color(hsv_image, bounds, i)
            
            if patch is not None:
                self.tracked_objects[marker_name].pixel_x = patch.centroid_x
                self.tracked_objects[marker_name].pixel_y = patch.centroid_y
                self.tracked_objects[marker_name].extra["area"] = patch.area
                self.tracked_objects[marker_name].extra["color_match_percentage"] = patch.color_match_percentage
                self.tracked_objects[marker_name].extra["detected"] = True
        
        # Annotate the image
        self.annotated_image = self.annotate_image(
            image=image, tracked_objects=self.tracked_objects
        )
        
        return self.tracked_objects

    def annotate_image(
        self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs
    ) -> np.ndarray:
        """
        Annotate the image with detected color markers.
        
        Args:
            image: Original image
            tracked_objects: Dictionary of tracked objects
            
        Returns:
            Annotated image
        """
        annotated_image = image.copy()
        
        # Draw each detected marker with its color
        for marker_name, tracked_object in tracked_objects.items():
            if tracked_object.pixel_x is not None and tracked_object.pixel_y is not None:
                # Find the marker config to get its color
                marker_config = None
                for bounds in self.hsv_bounds:
                    if bounds['config'].marker_name == marker_name:
                        marker_config = bounds['config']
                        marker_color = bounds['target_color_bgr']
                        break
                
                if marker_config is None:
                    continue
                
                # Draw a colored circle around the marker
                circle_color = tuple(int(c) for c in marker_color)
                cv2.circle(
                    img=annotated_image,
                    center=(int(tracked_object.pixel_x), int(tracked_object.pixel_y)),
                    radius=20,
                    color=circle_color,
                    thickness=3,
                )
                
                # Draw a cross at the centroid
                cv2.drawMarker(
                    img=annotated_image,
                    position=(int(tracked_object.pixel_x), int(tracked_object.pixel_y)),
                    color=(255, 255, 255),  # White cross
                    markerType=cv2.MARKER_CROSS,
                    markerSize=30,
                    thickness=2,
                )
                
                # Add marker name and info
                info_text = f"{marker_name}"
                if "area" in tracked_object.extra:
                    info_text += f" A:{tracked_object.extra['area']:.0f}"
                if "color_match_percentage" in tracked_object.extra:
                    info_text += f" M:{tracked_object.extra['color_match_percentage']:.0f}%"
                
                cv2.putText(
                    annotated_image,
                    info_text,
                    (int(tracked_object.pixel_x) + 25, int(tracked_object.pixel_y) - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )
        
        # Add legend with all configured markers (enabled and disabled)
        legend_y = 30
        for i, bounds in enumerate(self.hsv_bounds):
            config = bounds['config']
            status = "✓" if config.enabled else "✗"
            
            # Draw color swatch
            color_swatch = np.zeros((20, 20, 3), dtype=np.uint8)
            color_swatch[:, :] = bounds['target_color_bgr']
            annotated_image[legend_y:legend_y+20, 10:30] = color_swatch
            
            # Add text
            text = f"{status} {config.marker_name} ({config.target_color_bgr})"
            cv2.putText(
                annotated_image,
                text,
                (40, legend_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            
            legend_y += 25
        
        return annotated_image

    def update_marker_config(
        self, 
        marker_index: int, 
        enabled: Optional[bool] = None,
        target_color_bgr: Optional[Tuple[int, int, int]] = None,
        hue_tolerance: Optional[int] = None,
        saturation_tolerance: Optional[int] = None,
        value_tolerance: Optional[int] = None,
        marker_name: Optional[str] = None,
        min_contour_area: Optional[int] = None,
    ) -> None:
        """
        Update configuration for a specific marker.
        
        Args:
            marker_index: Index of marker to update (0-2)
            enabled: Whether the marker is enabled
            target_color_bgr: New target color in BGR
            hue_tolerance: New hue tolerance
            saturation_tolerance: New saturation tolerance
            value_tolerance: New value tolerance
            marker_name: New marker name
            min_contour_area: New minimum contour area
        """
        if marker_index >= len(self.marker_configs):
            logger.error(f"Marker index {marker_index} out of range")
            return
        
        config = self.marker_configs[marker_index]
        
        # Update configuration
        if enabled is not None:
            config.enabled = enabled
        
        if target_color_bgr is not None:
            config.target_color_bgr = target_color_bgr
        
        if hue_tolerance is not None:
            config.hue_tolerance = hue_tolerance

        if saturation_tolerance is not None:
            config.saturation_tolerance = saturation_tolerance

        if value_tolerance is not None:
            config.value_tolerance = value_tolerance
        
        if marker_name is not None:
            config.marker_name = marker_name
        
        if min_contour_area is not None:
            config.min_contour_area = min_contour_area
        
        # Re-initialize with updated configs
        self.__init__(
            marker_configs=self.marker_configs,
            use_morphological_ops=self.use_morphological_ops,
        )
        
        logger.info(f"Updated marker {marker_index}: {config.marker_name}")

    def get_marker_configs(self) -> List[ColorMarkerConfig]:
        """Get current marker configurations."""
        return self.marker_configs

    def enable_all_markers(self) -> None:
        """Enable all markers."""
        for config in self.marker_configs:
            config.enabled = True
        self.__init__(
            marker_configs=self.marker_configs,
            use_morphological_ops=self.use_morphological_ops,
        )

    def disable_all_markers(self) -> None:
        """Disable all markers."""
        for config in self.marker_configs:
            config.enabled = False
        self.__init__(
            marker_configs=self.marker_configs,
            use_morphological_ops=self.use_morphological_ops,
        )


if __name__ == "__main__":
    # Example: Track red and green markers
    marker_configs = [
        ColorMarkerConfig(
            enabled=True,
            target_color_bgr=(0, 0, 255),  # Red
            hue_tolerance=20,
            saturation_tolerance=70,
            value_tolerance=70,
            marker_name="red_marker",
            min_contour_area=100,
        ),
        ColorMarkerConfig(
            enabled=True,
            target_color_bgr=(0, 255, 0),  # Green
            hue_tolerance=20,
            saturation_tolerance=70,
            value_tolerance=70,
            marker_name="green_marker",
            min_contour_area=80,
        ),
        ColorMarkerConfig(
            enabled=False,  # Disabled blue marker
            target_color_bgr=(255, 0, 0),
            hue_tolerance=20,
            saturation_tolerance=70,
            value_tolerance=70,
            marker_name="blue_marker",
            min_contour_area=100,
        ),
    ]
    
    tracker = ColorTracker(marker_configs=marker_configs)
    tracker.demo()


__all__ = ["ColorTracker", "ColorMarkerConfig", "ColorPatch"]