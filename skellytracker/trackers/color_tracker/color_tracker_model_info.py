from typing import List

from skellytracker.trackers.base_tracker.model_info import ModelInfo
from skellytracker.trackers.color_tracker.color_tracker import ColorTracker, ColorMarkerConfig


class ColorTrackerModelInfo(ModelInfo):
    """Model info for color tracker."""
    
    # Class attributes (following MediapipeModelInfo pattern)
    name = "color_tracker"
    tracker_name = "ColorTracker"
    tracker = ColorTracker  # Note: This is the class, not an instance
    model_name = "color_tracker"
    
    # These will be set dynamically based on marker configs
    landmark_names = []  # Will be set in __init__
    num_tracked_points = 0  # Will be set in __init__
    tracked_object_names = []  # Will be set in __init__
    
    # Custom attributes for color tracker
    marker_configs = []
    use_morphological_ops = True
    min_circularity = 0.5
    min_solidity = 0.5
    
    def __init__(
        self,
        marker_configs: List[ColorMarkerConfig],
        use_morphological_ops: bool = True,
        min_circularity: float = 0.5,
        min_solidity: float = 0.5,
    ):
        # Get enabled marker names
        enabled_marker_names = [
            config.marker_name for config in marker_configs if config.enabled
        ]
        
        # Set instance attributes (which override class attributes)
        self.landmark_names = enabled_marker_names
        self.num_tracked_points = len(enabled_marker_names)
        self.tracked_object_names = enabled_marker_names
        self.marker_configs = marker_configs
        self.use_morphological_ops = use_morphological_ops
        self.min_circularity = min_circularity
        self.min_solidity = min_solidity
