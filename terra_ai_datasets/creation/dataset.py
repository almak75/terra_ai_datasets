import os
from typing import Union, List, Dict

import pandas as pd

from terra_ai_datasets.creation import creation_data
from terra_ai_datasets.creation.creation_data import InputData, OutputData, CSVData, InputInstructionsData, \
    OutputInstructionsData
from terra_ai_datasets.creation.validators import dataset
from terra_ai_datasets.creation.validators import inputs, outputs
from terra_ai_datasets.creation.validators.tasks import LayerInputTypeChoice, LayerOutputTypeChoice, \
    LayerSelectTypeChoice


class CreateDataset:
    data = None
    dataframe: pd.DataFrame = None
    input_type: LayerInputTypeChoice = None
    output_type: LayerOutputTypeChoice = None

    _is_validated = False
    _is_prepared = False
    _is_created = False

    def __init__(self, **kwargs):
        self.data = self.validate(
            getattr(dataset, f"{self.input_type}{self.output_type}Validator"), **kwargs
        )
        self.input, self.output = self.preprocess_put_data(
            data=self.data, data_type=LayerSelectTypeChoice.table
            if self.input_type == LayerInputTypeChoice.Dataframe else LayerSelectTypeChoice.folder
        )
        self.input_instructions = self.create_put_instructions(
            put_data=self.input, put_type='Input'
        )
        self.output_instructions = self.create_put_instructions(
            put_data=self.output, put_type='Output', start_idx=len(self.input_instructions) + 1
        )
        self._is_prepared = True

    def validate(self, instance, **kwargs):
        data = instance(**kwargs)
        self._is_validated = True
        return data

    def preprocess_put_data(self, data, data_type: LayerSelectTypeChoice):
        inputs_data = []
        outputs_data = []
        if data_type == LayerSelectTypeChoice.table:
            for inp in data.inputs:
                inputs_data.append(InputData(
                    csv_data=CSVData(csv_path=data.csv_path, columns=inp.columns),
                    type=inp.type,
                    parameters=inp.parameters
                ))
            for out in data.outputs:
                outputs_data.append(OutputData(
                    csv_data=CSVData(csv_path=data.csv_path, columns=out.columns),
                    type=out.type,
                    parameters=out.parameters
                ))
        elif data_type == LayerSelectTypeChoice.folder:
            inputs_data.append(InputData(
                folder_path=data.source_path,
                type=self.input_type,
                parameters=getattr(inputs, f"{self.input_type.value}Validator")(**data.dict())
            ))
            outputs_data.append(OutputData(
                folder_path=data.source_path,
                type=self.output_type,
                parameters=getattr(outputs, f"{self.output_type.value}Validator")(**data.dict())
            ))

        return inputs_data, outputs_data

    @staticmethod
    def create_put_instructions(put_data: List[Union[InputData, OutputData]], put_type: str, start_idx: int = 1) -> \
            Dict[int, Union[InputInstructionsData, OutputInstructionsData]]:

        new_put_data = {}
        for idx, put in enumerate(put_data, start_idx):
            columns = {}
            if put.csv_data:
                csv_table = pd.read_csv(put.csv_data.csv_path, usecols=put.csv_data.columns)
                for column in put.csv_data.columns:
                    data_to_pass = csv_table.loc[:, column].tolist()
                    columns[f"{idx}_{column}"] = data_to_pass

            elif put.folder_path:
                data_to_pass = []
                for folder_path in put.folder_path:
                    for direct, folder, files_name in os.walk(folder_path):
                        if files_name:
                            for file_name in sorted(files_name):
                                data_to_pass.append(folder_path.joinpath(file_name))
                columns[f"{idx}_{put.type.value}"] = data_to_pass

            new_put_data[idx] = getattr(creation_data, f"{put_type}InstructionsData")(
                type=put.type, parameters=put.parameters, columns=columns
            )
        return new_put_data

    def create(self):
        pass

    def summary(self):
        if not self._is_validated:
            raise
        print(self.dataframe.head())

    def load(self):
        pass

    def save(self):
        pass
