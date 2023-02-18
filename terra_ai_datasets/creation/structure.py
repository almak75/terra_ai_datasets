from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Union

from pydantic import BaseModel
from pydantic.color import Color
from pydantic.types import PositiveInt

from terra_ai_datasets.creation.validators.dataset import ImageClassificationValidator, ImageSegmentationValidator, \
    TextClassificationValidator
from terra_ai_datasets.creation.validators.tasks import TasksChoice, LayerDatatypeChoice, LayerEncodingChoice, \
    LayerInputTypeChoice, LayerOutputTypeChoice


class DatasetLayerData(BaseModel):
    name: str
    datatype: LayerDatatypeChoice
    dtype: str
    shape: Tuple[PositiveInt, ...]
    num_classes: Optional[PositiveInt]
    classes_names: Optional[List[str]]
    classes_colors: Optional[List[Color]]
    encoding: LayerEncodingChoice


class DatasetInputsData(DatasetLayerData):
    input_type: LayerInputTypeChoice


class DatasetOutputsData(DatasetLayerData):
    output_type: LayerOutputTypeChoice


class CreationData(BaseModel):
    inputs: Dict[PositiveInt, Union[ImageClassificationValidator, ImageSegmentationValidator,
                 TextClassificationValidator]]
    outputs: Dict[PositiveInt, Union[ImageClassificationValidator, ImageSegmentationValidator, TextClassificationValidator]]


class DatasetData(BaseModel):
    task: TasksChoice
    use_generator: bool
    date: datetime = datetime.now().isoformat()
    inputs: Optional[Dict[PositiveInt, DatasetInputsData]] = {}
    outputs: Optional[Dict[PositiveInt, DatasetOutputsData]] = {}
    columns: Optional[Dict[PositiveInt, Dict[str, Any]]] = {}
