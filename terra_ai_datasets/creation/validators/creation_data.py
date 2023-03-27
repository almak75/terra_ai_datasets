from typing import Optional, List, Any, Dict

from pydantic import BaseModel, PositiveInt, DirectoryPath, FilePath

from terra_ai_datasets.creation.validators.tasks import LayerInputTypeChoice, LayerOutputTypeChoice


# class CSVData(BaseModel):
#     csv_path: FilePath
#     columns: List[str]


class PutData(BaseModel):
    csv_path: Optional[FilePath]
    column: Optional[str]
    folder_path: Optional[List[DirectoryPath]]
    parameters: Any


class InputData(PutData):
    type: LayerInputTypeChoice

    # @validator("parameters")
    # def validate_input_parameters(cls, parameters, values):
    #     return getattr(inputs, f'{values["type"].value}Validator')(**parameters)


class OutputData(PutData):
    type: LayerOutputTypeChoice

    # @validator("parameters")
    # def validate_output_parameters(cls, parameters, values):
    #     return getattr(outputs, f'{values["type"].value}Validator')(**parameters)


class BaseInstructionsData(BaseModel):
    parameters: Any
    data: Optional[List[Any]]


class InputInstructionsData(BaseInstructionsData):
    type: LayerInputTypeChoice


class OutputInstructionsData(BaseInstructionsData):
    type: LayerOutputTypeChoice


class DatasetInstructionsData(BaseModel):
    inputs: Dict[PositiveInt, InputInstructionsData]
    outputs: Dict[PositiveInt, OutputInstructionsData]
