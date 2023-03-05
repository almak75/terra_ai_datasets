import json
from pathlib import Path
from datetime import datetime
from typing import Union, List, Dict, Tuple, Any

from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
from tensorflow import TensorSpec

from terra_ai_datasets.creation import arrays, preprocessings, utils
from terra_ai_datasets.creation.validators import creation_data
from terra_ai_datasets.creation.validators.creation_data import InputData, OutputData, CSVData, InputInstructionsData, \
    OutputInstructionsData
from terra_ai_datasets.creation.validators import dataset
from terra_ai_datasets.creation.validators import inputs, outputs
from terra_ai_datasets.creation.validators.structure import DatasetData

from terra_ai_datasets.creation.validators.tasks import LayerInputTypeChoice, LayerOutputTypeChoice, \
    LayerSelectTypeChoice


class TerraDataset:
    input_type: LayerInputTypeChoice = None
    output_type: LayerOutputTypeChoice = None
    X: Dict[str, Dict[int, np.ndarray]] = {"train": {}, "val": {}}
    Y: Dict[str, Dict[int, np.ndarray]] = {"train": {}, "val": {}}
    preprocessing: Dict[int, Any] = {}
    dataframe: Dict[str, pd.DataFrame] = {}
    dataset_data: DatasetData
    input: List[InputData] = []
    output: List[OutputData] = []

    _dataset: Dict[str, Dataset] = {"train": None, "val": None}
    _is_prepared: bool = False
    _is_created: bool = False

    @property
    def dataset(self):
        if not self._dataset["train"] and not self._dataset["val"]:
            return 'Arrays has not been created yet. call .create(use_generator: bool = False)'
        return self._dataset

    @staticmethod
    def create_dataset_object_from_arrays(x_arrays: Dict[int, np.ndarray], y_arrays: Dict[int, np.ndarray]) -> Dataset:
        return Dataset.from_tensor_slices((x_arrays, y_arrays))

    def create_dataset_object_from_instructions(self, inp_instr, out_instr, dataframe) -> Dataset:

        output_signature = [{}, {}]
        for put_id, p_data in enumerate(inp_instr + out_instr, 1):

            columns = [col for col in dataframe.columns.tolist() if col.startswith(str(put_id))]

            row_array = []
            for col_name in columns:
                sample_array = self.create_put_array(dataframe.loc[0, col_name], p_data)
                if self.preprocessing[put_id]:
                    sample_array = self.preprocess_put_array(np.array(sample_array), p_data,
                                                             self.preprocessing[put_id])
                row_array.append(sample_array)

            row_array = row_array if len(row_array) > 1 else row_array[0]
            if p_data.type in LayerInputTypeChoice:
                output_signature[0][put_id] = TensorSpec(shape=row_array.shape, dtype=row_array.dtype)
            else:
                output_signature[1][put_id] = TensorSpec(shape=row_array.shape, dtype=row_array.dtype)

        return Dataset.from_generator(lambda: self.generator(inp_instr, out_instr, dataframe),
                                      output_signature=tuple(output_signature))

    def generator(self, inp_instr, out_instr, dataframe: pd.DataFrame):

        for i in range(len(dataframe)):
            inp_dict, out_dict = {}, {}
            for put_id, p_data in enumerate(inp_instr + out_instr, 1):
                columns = [col for col in dataframe.columns.tolist() if col.startswith(str(put_id))]

                row_array = []
                for col_name in columns:
                    sample_array = self.create_put_array(dataframe.loc[0, col_name], p_data)
                    if self.preprocessing[put_id]:
                        sample_array = self.preprocess_put_array(
                            np.array(sample_array), p_data, self.preprocessing[put_id]
                        )
                    row_array.append(sample_array)
                row_array = row_array if len(row_array) > 1 else row_array[0]
                if p_data.type in LayerInputTypeChoice:
                    inp_dict[put_id] = row_array
                else:
                    out_dict[put_id] = row_array

            yield inp_dict, out_dict

    def create(self, use_generator: bool = False):

        def create_preprocessing(p_data: Union[InputData, OutputData]):
            preproc = None
            if "preprocessing" in p_data.parameters.dict() and p_data.parameters.preprocessing.value != 'None':
                preproc = \
                    getattr(preprocessings, f"create_{p_data.parameters.preprocessing.name}")(
                        p_data.parameters
                    )
            return preproc

        # def create_arrays_by_instructions(put_instructions: Union[InputData, OutputData], spl: str):
        #     arr_data = {}
        #     preproc_data = {}
        #     for put_id, p_data in put_instructions.items():
        #         preprocessing = create_preprocessing(p_data) if spl == "train" else self.preprocessing[put_id]
        #         array, preprocessing = self.create_arrays(
        #             p_data, dataframe, put_id, spl, preprocessing if spl == "train" else None
        #         )
        #         arr_data[put_id] = array
        #         if split == "train":
        #             preproc_data[put_id] = preprocessing
        #         preprocessing = preprocessing if preprocessing else self.preprocessing.get(put_id)
        #         if preprocessing:  # Временное решение до генераторов
        #             arr_data[put_id] = getattr(arrays, f"{p_data.type}Array")().preprocess(
        #                     arr_data[put_id], preprocessing, p_data.parameters
        #                 )
        #
        #     self._is_created = True
        #
        #     return arr_data, preproc_data

        if self._is_prepared:

            for put_id, p_data in enumerate(self.input + self.output, 1):
                self.preprocessing.update({put_id: create_preprocessing(p_data)})

            for split, dataframe in self.dataframe.items():  # train, val
                for put_id, p_data in enumerate(self.input + self.output, 1):
                    if split == "train":
                        self.preprocessing.update({put_id: create_preprocessing(p_data)})

                    columns = [col for col in dataframe.columns.tolist() if col.startswith(str(put_id))]
                    put_array = []
                    for row_idx in tqdm(range(len(dataframe)), desc=f"{datetime.now().strftime('%H:%M:%S')} | Формирование массивов {split} - {put_id} - {p_data.type}"):
                        row_array = []
                        for col_name in columns:
                            sample_array = self.create_put_array(dataframe.loc[row_idx, col_name], p_data)
                            if self.preprocessing[put_id] and split == "train":
                                if p_data.type == LayerInputTypeChoice.Text:
                                    self.preprocessing[put_id].fit_on_texts(sample_array.split())
                                else:
                                    self.preprocessing[put_id].fit(sample_array.reshape(-1, 1))
                            if not use_generator:
                                row_array.append(sample_array)
                        if not use_generator:
                            put_array.append(row_array if len(row_array) > 1 else row_array[0])

                    if not use_generator:
                        if self.preprocessing[put_id]:
                            put_array = self.preprocess_put_array(
                                np.array(put_array), p_data, self.preprocessing[put_id]
                            )
                        if isinstance(p_data, InputData):
                            self.X[split][put_id] = np.array(put_array)
                        else:
                            self.Y[split][put_id] = np.array(put_array)
                if use_generator and split == "train":
                    break

            for split, dataframe in self.dataframe.items():  # train, val
                if not use_generator:
                    self._dataset[split] = self.create_dataset_object_from_arrays(self.X[split], self.Y[split])
                else:
                    self._dataset[split] = self.create_dataset_object_from_instructions(
                        self.input, self.output, dataframe
                    )
        else:
            return 'Dataset was not prepared.'

    # @staticmethod
    # def create_arrays(put_data: Union[InputData, OutputData],
    #                   dataframe: pd.DataFrame,
    #                   put_id: int,
    #                   split: str,
    #                   preprocessing: Any = None,
    #                   ) -> Tuple[np.ndarray, Any]:
    #     row_array = []
    #     for row_idx in tqdm(
    #             range(len(dataframe)),
    #             desc=f"{datetime.now().strftime('%H:%M:%S')} | "
    #                  f"Формирование массивов {split} - {put_id} - {put_data.type}"
    #     ):
    #         for col_name in put_data.columns.keys():
    #             sample_array = getattr(arrays, f"{put_data.type}Array")().create(dataframe.loc[row_idx, col_name],
    #                                                                              put_data.parameters)
    #             if preprocessing:
    #                 if put_data.type == LayerInputTypeChoice.Text:
    #                     preprocessing.fit_on_texts(sample_array.split())
    #                 else:
    #                     preprocessing.fit(sample_array.reshape(-1, 1))
    #             row_array.append(sample_array)
    #
    #     return np.array(row_array), preprocessing

    @staticmethod
    def create_put_array(data: Any, put_data: Union[InputData, OutputData]):

        sample_array = getattr(arrays, f"{put_data.type}Array")().create(data, put_data.parameters)

        return sample_array

    @staticmethod
    def preprocess_put_array(data: Any, put_data: Union[InputData, OutputData], preprocessing: Any):

        preprocessed_array = getattr(arrays, f"{put_data.type}Array")().preprocess(
            data, preprocessing, put_data.parameters
        )

        return preprocessed_array

    def summary(self):
        if not self._is_prepared:
            raise

        print(self.dataframe['train'].head())
        print(f"\n\033[1mКол-во примеров в train выборке:\033[0m {len(self.dataframe['train'])}\n"
              f"\033[1mКол-во примеров в val выборке:\033[0m {len(self.dataframe['val'])}")
        print()
        if self._is_created:
            for inp_id, array in enumerate(self.X["train"].values(), 1):
                print(f"\033[1mРазмерность входного массива {inp_id}:\033[0m", array[0].shape)
            for out_id, array in enumerate(self.Y["train"].values(), 1):
                print(f"\033[1mРазмерность выходного массива {out_id}:\033[0m", array[0].shape)

    def save(self, save_path: str) -> None:

        def arrays_save(arrays_data: Dict[str, Dict[int, np.ndarray]], path_to_folder: Path):
            for split, data in arrays_data.items():
                for put_id, array in data.items():
                    np.save(str(path_to_folder.joinpath(f"{put_id}_{split}")), array)

        path_to_save = Path(save_path)
        dataset_paths_data = utils.DatasetPathsData(path_to_save)

        arrays_save(self.X, dataset_paths_data.arrays.inputs)
        arrays_save(self.Y, dataset_paths_data.arrays.outputs)
        for split, dataframe in self.dataframe.items():
            dataframe.to_csv(dataset_paths_data.instructions.dataframe.joinpath(f"{split}.csv"))

        for idx, inp in enumerate(self.input, 1):
            with open(dataset_paths_data.instructions.parameters.joinpath(f"input_{idx}_{inp.type}.json"), "w")\
                    as instruction:
                json.dump(inp.json(), instruction)

        for idx, out in enumerate(self.output, 1 + idx):
            with open(dataset_paths_data.instructions.parameters.joinpath(f"output_{idx}_{out.type}.json"), "w")\
                    as instruction:
                json.dump(out.json(), instruction)

        with open(dataset_paths_data.config, "w") as config:
            json.dump(self.dataset_data.json(), config)
        if not path_to_save.is_absolute():
            path_to_save = Path.cwd().joinpath(path_to_save)

        utils.logger.info(f"Датасет сохранен в директорию {path_to_save}")


class CreateDataset(TerraDataset):

    _is_created: bool = False

    def __init__(self, **kwargs):
        data = self._validate(  # убрать из self
            getattr(dataset, f"{self.input_type}{self.output_type}Validator"), **kwargs
        )
        self.input, self.output = self.preprocess_put_data(
            data=data, data_type=LayerSelectTypeChoice.table
            if self.input_type == LayerInputTypeChoice.Dataframe else LayerSelectTypeChoice.folder
        )
        input_instructions = self.create_put_instructions(
            put_data=self.input, put_type='Input'
        )
        output_instructions = self.create_put_instructions(
            put_data=self.output, put_type='Output', start_idx=len(input_instructions) + 1
        )
        self.dataframe = self.create_table(
            input_instructions, output_instructions, train_size=data.train_size
        )

        self.dataset_data = DatasetData(
            task=self.input_type.value + self.output_type.value,
            use_generator=data.use_generator,
            is_created=self._is_created,
        )
        self._is_prepared = True

        utils.logger.info(f"Датасет подготовлен к началу формирования массивов")

    @staticmethod
    def _validate(instance, **kwargs):
        data = instance(**kwargs)
        return data

    def preprocess_put_data(self, data, data_type: LayerSelectTypeChoice) -> Tuple[List[InputData], List[OutputData]]:
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
                    if put.type in [LayerInputTypeChoice.Image, LayerOutputTypeChoice.Segmentation]:
                        data_to_pass = [put.csv_data.csv_path.parent.joinpath(elem) for elem in data_to_pass]
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
            dataframe, [int(train_size * len(dataframe))]
        )
        dataframe = {"train": train_dataframe.reset_index(drop=True), "val": val_dataframe.reset_index(drop=True)}

        return dataframe


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

    def summary(self):
        super().summary()
        print(f"\n\033[1mСписок классов:\033[0m", ' '.join(self.output[0].parameters.classes_names))
