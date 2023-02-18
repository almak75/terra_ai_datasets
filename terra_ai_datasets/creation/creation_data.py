from typing import Optional, List, Any, Dict
from pathlib import Path

from pydantic import BaseModel, validator, PositiveInt

# from terra_ai_datasets.creation.validators import inputs, outputs
from terra_ai_datasets.creation.validators.tasks import LayerInputTypeChoice, LayerOutputTypeChoice


class CSVData(BaseModel):
    csv_path: Path
    columns: List[str]


class PutData(BaseModel):
    csv_data: Optional[CSVData]
    folder_path: Optional[List[Path]]
    parameters: dict


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


# class InstructionsData(BaseModel):
#     instructions: List


class BaseInstructionsData(BaseModel):
    parameters: dict
    preprocess: dict = {}
    columns: Dict[str, List]


class InputInstructionsData(BaseInstructionsData):
    type: LayerInputTypeChoice


class OutputInstructionsData(BaseInstructionsData):
    type: LayerOutputTypeChoice


class DatasetInstructionsData(BaseModel):
    inputs: Dict[PositiveInt, InputInstructionsData]
    outputs: Dict[PositiveInt, OutputInstructionsData]
