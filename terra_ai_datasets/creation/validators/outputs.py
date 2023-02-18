from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel
from pydantic.types import PositiveInt
from pydantic.color import Color


class RegressionScalers(str, Enum):
    none = None
    min_max_scaler = "MinMaxScaler"
    terra_image_scaler = "StandardScaler"


class ClassificationValidator(BaseModel):
    one_hot_encoding: bool = True


class SegmentationValidator(BaseModel):
    rgb_range: PositiveInt
    classes: Dict[str, Color]


class RegressionValidator(BaseModel):
    regression_scaler: RegressionScalers = RegressionScalers.none
