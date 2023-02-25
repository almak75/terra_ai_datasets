import os
import json
from collections import ChainMap
from datetime import datetime
from pathlib import Path
from typing import Union, List, Dict, Any, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd

from terra_ai_datasets.creation import arrays, preprocessings, utils
from terra_ai_datasets.creation.utils import DatasetPathsData, logger
from terra_ai_datasets.creation.validators import creation_data
from terra_ai_datasets.creation.validators.creation_data import InputData, OutputData, CSVData, InputInstructionsData, \
    OutputInstructionsData
from terra_ai_datasets.creation.validators import dataset
from terra_ai_datasets.creation.validators import inputs, outputs
from terra_ai_datasets.creation.validators.structure import DatasetData
from terra_ai_datasets.creation.validators.tasks import LayerInputTypeChoice, LayerOutputTypeChoice, \
    LayerSelectTypeChoice


class CreateDataset:
    data = None
    input_type: LayerInputTypeChoice = None
    output_type: LayerOutputTypeChoice = None
    dataframe: Dict[str, pd.DataFrame] = {}
    preprocessing: Dict[int, Any] = {}
    X: Dict[str, Dict[int, np.ndarray]] = {}
    Y: Dict[str, Dict[int, np.ndarray]] = {}

    _is_validated = False
    _is_prepared = False
    _is_created = False

    def __init__(self, **kwargs):
        self.data = self._validate(
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
        self.dataframe = self.create_table(
            self.input_instructions, self.output_instructions, train_size=self.data.train_size
        )
        self._is_prepared = True
        logger.info(f"Датасет подготовлен к началу формирования массивов")

    def _validate(self, instance, **kwargs):
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
                    data_to_pass.extend(
                        getattr(utils, f"extract_{put.type.value.lower()}_data")(folder_path, put.parameters)
                    )
                columns[f"{idx}_{put.type.value}"] = data_to_pass

            new_put_data[idx] = getattr(creation_data, f"{put_type}InstructionsData")(
                type=put.type, parameters=put.parameters, columns=columns
            )
        return new_put_data

    @staticmethod
    def create_table(input_instructions: Dict[int, InputInstructionsData],
                     output_instructions: Dict[int, OutputInstructionsData],
                     train_size: int,
                     shuffle: bool = True,
                     ) -> Dict[str, pd.DataFrame]:

        def create_put_table(put_instructions):
            dict_data = {}
            for put_id, put_data in put_instructions.items():
                dict_data.update(put_data.columns)
            return dict_data

        csv_data = {}
        csv_data.update(create_put_table(input_instructions))
        csv_data.update(create_put_table(output_instructions))
        dataframe = pd.DataFrame.from_dict(csv_data)

        if shuffle:
            dataframe = dataframe.sample(frac=1)

        train_dataframe, val_dataframe = np.split(
            dataframe, [int(train_size / 100 * len(dataframe))]
        )
        dataframe = {"train": train_dataframe.reset_index(drop=True), "val": val_dataframe.reset_index(drop=True)}

        return dataframe

    def create(self):

        def create_preprocessing(put_data):
            preproc = None
            if "preprocessing" in put_data.parameters.dict() and put_data.parameters.preprocessing.value != 'None':
                preproc = \
                    getattr(preprocessings, f"create_{put_data.parameters.preprocessing.name}")(
                        put_data.parameters
                    )
            return preproc

        def create_arrays_by_instructions(put_instructions, spl: str):
            arr_data = {}
            preproc_data = {}
            for put_id, put_data in put_instructions.items():
                preprocessing = create_preprocessing(put_data) if spl == "train" else self.preprocessing[put_id]
                array, preprocessing = self.create_arrays(
                    put_data, dataframe, put_id, spl, preprocessing if spl == "train" else None
                )
                arr_data[put_id] = array
                if split == "train":
                    preproc_data[put_id] = preprocessing
                preprocessing = preprocessing if preprocessing else self.preprocessing.get(put_id)
                if preprocessing:  # Временное решение до генераторов
                    arr_data[put_id] = getattr(arrays, f"{put_data.type}Array")().preprocess(
                            arr_data[put_id], preprocessing, put_data.parameters
                        )

            return arr_data, preproc_data

        if self._is_prepared:
            for split, dataframe in self.dataframe.items():

                arrays_data, preprocessing_data = create_arrays_by_instructions(self.input_instructions, split)
                self.X[split] = arrays_data
                self.preprocessing.update(preprocessing_data)

                arrays_data, preprocessing_data = create_arrays_by_instructions(self.output_instructions, split)
                self.Y[split] = arrays_data
                self.preprocessing.update(preprocessing_data)

    @staticmethod
    def create_arrays(put_data: Union[InputInstructionsData, OutputInstructionsData],
                      dataframe: pd.DataFrame,
                      put_id: int,
                      split: str,
                      preprocessing: Any = None,
                      ) -> Tuple[np.ndarray, Any]:
        row_array = []
        for row_idx in tqdm(
                range(len(dataframe)),
                desc=f"{datetime.now().strftime('%H:%M:%S')} | "
                     f"Формирование массивов {split} - {put_id} - {put_data.type}"
        ):
            for col_name in put_data.columns.keys():
                sample_array = getattr(arrays, f"{put_data.type}Array")().create(dataframe.loc[row_idx, col_name],
                                                                                 put_data.parameters)
                if preprocessing:
                    if put_data.type == LayerInputTypeChoice.Text:
                        preprocessing.fit_on_texts(sample_array.split())
                    else:
                        preprocessing.fit(sample_array.reshape(-1, 1))
                row_array.append(sample_array)

        return np.array(row_array), preprocessing

    def summary(self):
        if not self._is_validated:
            raise
        if self._is_prepared:
            print(self.dataframe['train'].head())

    def save(self, save_path: str) -> None:

        def arrays_save(arrays_data: Dict[str, Dict[int, np.ndarray]], path_to_save: Path):
            for split, data in arrays_data.items():
                for put_id, array in data.items():
                    np.save(str(path_to_save.joinpath(f"{put_id}_{split}")), array)

        path_to_save = Path(save_path)
        dataset_paths_data = DatasetPathsData(path_to_save)

        arrays_save(self.X, dataset_paths_data.arrays.inputs)
        arrays_save(self.Y, dataset_paths_data.arrays.outputs)

        dataset_data = DatasetData(
            task=self.input_type.value + self.output_type.value,
            use_generator=self.data.use_generator
        )

        with open(dataset_paths_data.config, "w") as config:
            json.dump(dataset_data.dict(), config)
        if not path_to_save.is_absolute():
            path_to_save = Path.cwd().joinpath(path_to_save)

        logger.info(f"Датасет сохранен в директорию {path_to_save}")


class CreateClassificationDataset(CreateDataset):
    y_classes = []

    def create_put_instructions(self, put_data: List[Union[InputData, OutputData]], put_type: str, start_idx: int = 1) -> \
            Dict[int, Union[InputInstructionsData, OutputInstructionsData]]:

        new_put_data = {}
        for idx, put in enumerate(put_data, start_idx):
            columns = {}
            if put_type == "Input":
                data_to_pass = []
                for folder_path in put.folder_path:
                    data = getattr(utils, f"extract_{put.type.value.lower()}_data")(folder_path, put.parameters)
                    self.y_classes.extend([folder_path.name for _ in data])
                    data_to_pass.extend(data)
            else:
                data_to_pass = self.y_classes
                put.parameters.classes_names = [path.name for path in put.folder_path]
            columns[f"{idx}_{put.type.value}"] = data_to_pass

            new_put_data[idx] = getattr(creation_data, f"{put_type}InstructionsData")(
                type=put.type, parameters=put.parameters, columns=columns
            )
        return new_put_data
