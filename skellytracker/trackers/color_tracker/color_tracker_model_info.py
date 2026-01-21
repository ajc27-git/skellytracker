from typing import List
from pydantic import BaseModel

from skellytracker.trackers.base_tracker.model_info import ModelInfo
from skellytracker.trackers.color_tracker.color_tracker import ColorTracker, ColorMarkerConfig


class ColorTrackerModelInfo(ModelInfo):
    """Model info for color tracker."""
    
    def __init__(
        self,
        marker_configs: List[ColorMarkerConfig],
        use_morphological_ops: bool = True,
    ):
        super().__init__(
            tracker_name="ColorTracker",
            tracker=ColorTracker,  # The tracker class
            model_name="color_tracker",
            marker_configs=marker_configs,
            use_morphological_ops=use_morphological_ops,
        )
        self.marker_configs = marker_configs
        self.use_morphological_ops = use_morphological_ops