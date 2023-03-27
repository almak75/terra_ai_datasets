import errno
import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, validator

from terra_ai_datasets.creation.validators import inputs, outputs
from terra_ai_datasets.creation.validators.tasks import LayerInputTypeChoice, LayerOutputTypeChoice


# --- Common validators ---
class CommonValidator(BaseModel):
    train_size: float

    @validator('train_size')
    def train_size_range(cls, v):
        if not 0.0 < v <= 1.0:
            raise ValueError('Ensure train_size is in range between 0.0 and 1.0')
        return v

    class Config:
        use_enum_values = True


class SourceFolderPathValidator(CommonValidator):
    source_path: List[Path]

    @validator('source_path', each_item=True)
    def sources_folder_exists(cls, source_path: Path):
        if not source_path.is_dir():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), source_path
                )
        return source_path


class TargetFolderPathValidator(CommonValidator):
    target_path: List[Path]

    @validator('target_path', each_item=True)
    def target_folder_exists(cls, target_path: Path):
        if not target_path.is_dir():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), target_path
                )
        return target_path


class FilePathValidator(CommonValidator):
    csv_path: Path

    @validator('csv_path')
    def file_path_exists(cls, csv_path: Path):
        if not csv_path.is_file():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), csv_path
                )
        if not csv_path.suffix == '.csv':
            raise Exception('Must be a .csv file')
        return csv_path


# --- Image validators ---
class ImageClassificationValidator(SourceFolderPathValidator, inputs.ImageValidator, outputs.ClassificationValidator):
    pass


class ImageSegmentationValidator(SourceFolderPathValidator, TargetFolderPathValidator,
                                 inputs.ImageValidator, outputs.SegmentationValidator):
    pass


# --- Text validators ---
class TextClassificationValidator(SourceFolderPathValidator, inputs.TextValidator, outputs.ClassificationValidator):
    pass


# --- Audio validators ---
class AudioClassificationValidator(SourceFolderPathValidator, inputs.AudioValidator, outputs.ClassificationValidator):
    pass


# --- Dataframe Validators ---
class DataframePutData(BaseModel):
    columns: List[str]


class DataframeInputData(DataframePutData):
    type: LayerInputTypeChoice
    parameters: dict

    @validator("parameters")
    def validate_input_parameters(cls, parameters, values):
        if not values.get("type"):
            raise ValueError
        return getattr(inputs, f'{values["type"].value}Validator')(**parameters)


class DataframeOutputData(DataframePutData):
    type: LayerOutputTypeChoice
    parameters: dict

    @validator("parameters")
    def validate_output_parameters(cls, parameters, values):
        if not values.get("type"):
            raise ValueError
        return getattr(outputs, f'{values["type"].value}Validator')(**parameters)


class DataframeValidator(BaseModel):
    inputs: List[DataframeInputData]
    output: str


class DataframeClassificationValidator(FilePathValidator, DataframeValidator, outputs.ClassificationValidator):
    pass


class DataframeRegressionValidator(FilePathValidator, DataframeValidator, outputs.RegressionValidator):
    pass


class TimeseriesDepthValidator(FilePathValidator, inputs.TimeseriesValidator, outputs.DepthValidator):
    inputs: List[str]
    outputs: List[str]


class TimeseriesTrendValidator(FilePathValidator, inputs.TimeseriesValidator, outputs.TrendValidator):
    inputs: List[str]
    output: str
