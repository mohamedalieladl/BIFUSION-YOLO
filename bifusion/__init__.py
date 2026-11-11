"""BiFusion-YOLO: a lightweight P2-enhanced YOLO26 for ultra-small UAV detection."""

from .model import BiFusionYOLO, build_model, count_parameters

__all__ = ["BiFusionYOLO", "build_model", "count_parameters"]
__version__ = "1.0.0"
