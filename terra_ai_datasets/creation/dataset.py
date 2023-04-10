import json
from pathlib import Path
from datetime import datetime
from typing import Union, Dict, Any

from IPython.display import display
from tensorflow.python.data.ops.dataset_ops import DatasetV2 as Dataset
from tensorflow import TensorSpec
from tqdm import tqdm
import numpy as np
import pandas as pd
import joblib

from terra_ai_datasets.creation import preprocessings, utils, visualize
from terra_ai_datasets.creation.utils import apply_pymorphy, process_directory_paths, decamelize, create_put_array, \
    preprocess_put_array
from terra_ai_datasets.creation.validators import creation_data
from terra_ai_datasets.creation.validators.creation_data import InputData, OutputData, InputInstructionsData, \
    OutputInstructionsData
from terra_ai_datasets.creation.validators import dataset
from terra_ai_datasets.creation.validators import inputs, outputs
from terra_ai_datasets.creation.validators.inputs import CategoricalValidator, ImageScalers, TextProcessTypes
from terra_ai_datasets.creation.validators.structure import DatasetData

from terra_ai_datasets.creation.validators.tasks import LayerInputTypeChoice, LayerOutputTypeChoice, \
    LayerSelectTypeChoice


def postprocess_put_array(put_arr: list, put_data_type: Union[LayerInputTypeChoice, LayerOutputTypeChoice]) \
        -> np.ndarray:
    if put_data_type in [LayerInputTypeChoice.Timeseries, LayerOutputTypeChoice.Depth]:
        # put_arr = np.moveaxis(np.array(put_arr), [1, 2], [0, 1])
        put_arr = np.concatenate(put_arr, axis=2)
        pass
    # arrays
    elif all([len(np.array(arr).shape) >= 2 for arr in put_arr]):
        put_arr = np.concatenate(put_arr, axis=1)
    # vectors
    elif all([len(np.array(arr).shape) == 1 for arr in put_arr]):
        put_arr = np.concatenate(put_arr, axis=0)
    # scalar
    else:
        put_arr = np.array(put_arr[0])
    return put_arr


class TerraDataset:

    def __init__(self):
        self.X: Dict[str, Dict[str, np.ndarray]] = {"train": {}, "val": {}}
        self.Y: Dict[str, Dict[str, np.ndarray]] = {"train": {}, "val": {}}
        self.preprocessing: Dict[str, Any] = {}
        self.dataframe: Dict[str, pd.DataFrame] = {}
        self.dataset_data: DatasetData = None
        self.put_instructions: Dict[int, Dict[str, Union[InputInstructionsData, OutputInstructionsData]]] = {}

        self._dataset: Dict[str, Dataset] = {"train": None, "val": None}

    @property
    def dataset(self):
        if not self._dataset["train"] and not self._dataset["val"]:
            return 'Массивы не были созданы. Вызовите метод .create(use_generator: bool = False)'
        return self._dataset

    @staticmethod
    def create_dataset_object_from_arrays(x_arrays: Dict[str, np.ndarray], y_arrays: Dict[str, np.ndarray]) -> Dataset:
        return Dataset.from_tensor_slices((x_arrays, y_arrays))

    def create_dataset_object_from_instructions(self, put_instr, dataframe) -> Dataset:

        output_signature = [{}, {}]
        length, step, offset, total_samples = 1, 1, 0, len(dataframe)
        inp_id, out_id = 1, 1
        for put_id, cols_dict in put_instr.items():
            put_array = []
            for col_name, put_data in cols_dict.items():
                if put_data.type in \
                        [LayerInputTypeChoice.Timeseries, LayerOutputTypeChoice.Depth, LayerOutputTypeChoice.Trend]:
                    length = put_data.parameters.length
                    total_samples -= length * 2
                    if put_data.type in [LayerOutputTypeChoice.Depth, LayerOutputTypeChoice.Trend]:
                        offset = put_data.parameters.length
                        if put_data.type == LayerOutputTypeChoice.Trend:
                            offset -= 1
                data_to_send = dataframe.loc[0 + offset:0 + offset + length - 1, col_name].to_list()
                data_to_send = data_to_send[0] if len(data_to_send) == 1 else data_to_send
                sample_array = create_put_array(data_to_send, put_data)
                if self.preprocessing.get(col_name):
                    sample_array = preprocess_put_array(
                        sample_array, put_data, self.preprocessing[col_name]
                    )
                put_array.append(
                    sample_array[0] if type(sample_array) == np.ndarray and len(sample_array) == 1 else sample_array
                )
            if put_data.type in [LayerInputTypeChoice.Timeseries, LayerOutputTypeChoice.Depth]:
                put_array = np.expand_dims(np.array(put_array), 1)
                put_array = postprocess_put_array(put_array, put_data.type)[0]
            else:
                put_array = postprocess_put_array(put_array, put_data.type)

            if put_data.type in LayerInputTypeChoice:
                output_signature[0][f"input_{inp_id}"] = TensorSpec(shape=put_array.shape, dtype=put_array.dtype)
                inp_id += 1
            else:
                output_signature[1][f"output_{out_id}"] = TensorSpec(shape=put_array.shape, dtype=put_array.dtype)
                out_id += 1

        return Dataset.from_generator(lambda: self.generator(put_instr, dataframe, length, step, total_samples),
                                      output_signature=tuple(output_signature))

    def generator(self, put_instr, dataframe: pd.DataFrame, length, step, total_samples):

        for i in range(0, total_samples, step):
            inp_id, out_id = 1, 1
            inp_dict, out_dict = {}, {}
            for put_id, cols_dict in put_instr.items():
                put_array = []
                for col_name, put_data in cols_dict.items():
                    offset = 0
                    if put_data.type in [LayerOutputTypeChoice.Depth, LayerOutputTypeChoice.Trend]:
                        offset = put_data.parameters.length
                    data_to_send = dataframe.loc[i+offset: i+offset+length-1, col_name].to_list()
                    data_to_send = data_to_send[0] if len(data_to_send) == 1 else data_to_send
                    sample_array = create_put_array(data_to_send, put_data)

                    # sample_array = create_put_array(dataframe.loc[i, col_name], put_data)
                    if self.preprocessing.get(col_name):
                        sample_array = preprocess_put_array(
                            sample_array, put_data, self.preprocessing[col_name]
                        )
                    put_array.append(
                        sample_array[0] if type(sample_array) == np.ndarray and len(sample_array) == 1 else sample_array
                    )

                if put_data.type in [LayerInputTypeChoice.Timeseries, LayerOutputTypeChoice.Depth]:
                    put_array = np.expand_dims(np.array(put_array), 1)
                    put_array = postprocess_put_array(put_array, put_data.type)[0]
                else:
                    put_array = postprocess_put_array(put_array, put_data.type)

                if put_data.type in LayerInputTypeChoice:
                    inp_dict[f"input_{inp_id}"] = put_array
                    inp_id += 1
                else:
                    out_dict[f"output_{out_id}"] = put_array
                    out_id += 1

            yield inp_dict, out_dict

    def create(self, use_generator: bool = False):

        for split, dataframe in self.dataframe.items():
            inp_id, out_id = 1, 1
            for put_id, cols_dict in self.put_instructions.items():
                if use_generator and split != "train":
                    continue
                for col_name, put_data in cols_dict.items():
                    if split == "train":
                        if "preprocessing" in put_data.parameters.dict() and put_data.parameters.preprocessing.value != 'None':
                            create_preprocessing_parameters = {"parameters": put_data.parameters}
                            if put_data.parameters.preprocessing == TextProcessTypes.word_to_vec:
                                create_preprocessing_parameters.update(
                                    {"text_list": dataframe[col_name].to_list()}
                                )
                            self.preprocessing[col_name] = \
                                getattr(preprocessings, f"create_{put_data.parameters.preprocessing.name}")(
                                    **create_preprocessing_parameters
                                )

                put_array = []
                for col_name, put_data in cols_dict.items():
                    col_array = []
                    length, step, offset, total_samples = 1, 1, 0, len(dataframe)
                    if put_data.type in \
                            [LayerInputTypeChoice.Timeseries, LayerOutputTypeChoice.Depth, LayerOutputTypeChoice.Trend]:
                        length = put_data.parameters.length
                        step = put_data.parameters.step
                        total_samples -= length * 2
                        if put_data.type in [LayerOutputTypeChoice.Depth, LayerOutputTypeChoice.Trend]:
                            offset = put_data.parameters.length
                            if put_data.type == LayerOutputTypeChoice.Trend:
                                offset -= 1
                    for row_idx in tqdm(
                            range(0, total_samples, step),
                            desc=f"{datetime.now().strftime('%H:%M:%S')} | Формирование массивов {split} - {put_data.type} - {col_name}"
                    ):
                        data_to_send = dataframe.loc[row_idx+offset:row_idx+offset+length-1, col_name].to_list()
                        data_to_send = data_to_send[0] if len(data_to_send) == 1 else data_to_send
                        sample_array = create_put_array(data_to_send, put_data)
                        if self.preprocessing.get(col_name) and split == "train":
                            if put_data.type == LayerInputTypeChoice.Text:
                                pass
                            elif put_data.type == LayerInputTypeChoice.Image and put_data.parameters.preprocessing == ImageScalers.terra_image_scaler:
                                self.preprocessing[col_name].fit(sample_array)
                            else:
                                self.preprocessing[col_name].partial_fit(sample_array.reshape(-1, 1))
                        if not use_generator:
                            col_array.append(
                                sample_array[0] if type(sample_array) == np.ndarray and len(sample_array) == 1 else sample_array
                            )

                    if self.preprocessing.get(col_name) and split == "train":
                        if put_data.type == LayerInputTypeChoice.Text:
                            if put_data.parameters.preprocessing != TextProcessTypes.word_to_vec:
                                self.preprocessing[col_name].fit_on_texts(col_array)

                    if self.preprocessing.get(col_name) and not use_generator:
                        col_array = preprocess_put_array(
                            np.array(col_array), put_data, self.preprocessing[col_name]
                        )
                    if not use_generator:
                        put_array.append(col_array if len(col_array) > 1 else col_array[0])

                if not use_generator:
                    put_array = postprocess_put_array(put_array, put_data.type)
                    if isinstance(put_data, InputInstructionsData):
                        self.X[split][f"input_{inp_id}"] = put_array
                        inp_id += 1
                    else:
                        self.Y[split][f"output_{out_id}"] = put_array
                        out_id += 1

            if not use_generator:
                self._dataset[split] = self.create_dataset_object_from_arrays(self.X[split], self.Y[split])
            else:
                self._dataset[split] = self.create_dataset_object_from_instructions(
                    self.put_instructions, dataframe
                )
        self.dataset_data.is_created = True

    def visualize(self):
        getattr(visualize, f"visualize_{decamelize(self.dataset_data.task)}")(
            put_instructions=self.put_instructions,
            preprocessing=self.preprocessing
        )

    def evaluate_on_model(self, model, batch_size: int):
        for inp, out in self.dataset["val"].batch(batch_size):
            pred = model.predict(inp)
            yield inp, out, pred

    def summary(self):

        display(self.dataframe['train'].head())
        print(f"\n\033[1mКол-во примеров в train выборке:\033[0m {len(self.dataframe['train'])}\n"
              f"\033[1mКол-во примеров в val выборке:\033[0m {len(self.dataframe['val'])}")
        print()
        if self.dataset_data.is_created:
            for inp_id, array in enumerate(self.X["train"].values(), 1):
                print(f"\033[1mРазмерность входного массива {inp_id}:\033[0m", array[0].shape)
            for out_id, array in enumerate(self.Y["train"].values(), 1):
                print(f"\033[1mРазмерность выходного массива {out_id}:\033[0m", array[0].shape)

        put_id, col_name, output_type = None, None, None
        for put_id, put_data in self.put_instructions.items():
            for col_name, col_data in put_data.items():
                if col_data.type in LayerOutputTypeChoice:
                    output_type = col_data.type
                    break

        if output_type == LayerOutputTypeChoice.Classification:
            print("\033[1mСписок классов и количество примеров:\033[0m")
            classes_names = self.put_instructions[put_id][col_name].parameters.classes_names
            dataframe_dict = {cl_name: [] for cl_name in classes_names}
            for split in ['train', 'val']:
                list_of_elements = self.dataframe[split].loc[:, col_name].tolist()
                for cl_name in classes_names:
                    dataframe_dict[cl_name].append(list_of_elements.count(cl_name))

            display(pd.DataFrame(dataframe_dict, index=['train', 'val']))

        elif output_type == LayerOutputTypeChoice.Segmentation:
            text_to_print = "\033[1mКлассы в масках сегментации и их цвета в RGB:\033[0m"
            classes = self.put_instructions[put_id][col_name].parameters.classes
            for name, color in classes.items():
                text_to_print += f"\n{name}: {color.as_rgb_tuple()}"
            print(text_to_print)

    def save(self, save_path: str) -> None:

        def arrays_save(arrays_data: Dict[str, Dict[int, np.ndarray]], path_to_folder: Path):
            for spl, data in arrays_data.items():
                for p_id, array in data.items():
                    joblib.dump(array, path_to_folder.joinpath(f"{p_id}_{spl}.gz"))

        path_to_save = Path(save_path)
        dataset_paths_data = utils.DatasetPathsData(path_to_save)

        arrays_save(self.X, dataset_paths_data.arrays.inputs)
        arrays_save(self.Y, dataset_paths_data.arrays.outputs)

        if self.preprocessing:
            for col_name, proc in self.preprocessing.items():
                if proc:
                    joblib.dump(proc, dataset_paths_data.preprocessing.joinpath(f"{col_name}.gz"))

        for split, dataframe in self.dataframe.items():
            dataframe.to_csv(dataset_paths_data.instructions.dataframe.joinpath(f"{split}.csv"))

        for put_id, cols_dict in self.put_instructions.items():
            for col_name, put_data in cols_dict.items():
                put_type = "input" if put_data.type in LayerInputTypeChoice else "output"
                file_name = f"{put_type}_{put_id}_{put_data.type}"
                with open(dataset_paths_data.instructions.parameters.joinpath(f"{file_name}.json"), "w") as instruction:
                    put_data.data = None
                    json.dump(put_data.json(), instruction)

        with open(dataset_paths_data.config, "w") as config:
            json.dump(self.dataset_data.json(), config)
        if not path_to_save.is_absolute():
            path_to_save = Path.cwd().joinpath(path_to_save)

        utils.logger.info(f"Датасет сохранен в директорию {path_to_save}")


class CreateDataset(TerraDataset):
    input_type: LayerInputTypeChoice = None
    output_type: LayerOutputTypeChoice = None

    def __init__(self, **kwargs):
        super().__init__()
        self.data = self._validate(
            getattr(dataset, f"{self.input_type}{self.output_type}Validator"), **kwargs
        )
        self.put_data = self.preprocess_put_data(
            data=self.data, data_type=LayerSelectTypeChoice.table
            if self.input_type == LayerInputTypeChoice.Dataframe else LayerSelectTypeChoice.folder
        )
        self.put_instructions = self.create_put_instructions(put_data=self.put_data)
        self.dataframe = self.create_table(
            self.put_instructions,
            train_size=self.data.train_size,
            shuffle=True if self.input_type != LayerInputTypeChoice.Timeseries else False
        )

        self.dataset_data = DatasetData(
            task=self.input_type.value + self.output_type.value,
            use_generator=False,
            is_created=False,
        )

        utils.logger.info(f"Датасет подготовлен к началу формирования массивов")

    @staticmethod
    def _validate(instance, **kwargs):
        data = instance(**kwargs)
        return data

    def preprocess_put_data(self, data, data_type: LayerSelectTypeChoice) -> \
            Dict[int, Union[Dict[Any, Any], Dict[str, InputData], Dict[str, OutputData]]]:

        puts_data = {}

        if data_type == LayerSelectTypeChoice.table:
            for idx, put in enumerate(data.inputs, 1):
                puts_data[idx] = {}
                for col_name in put.columns:
                    parameters_to_pass = {"csv_path": data.csv_path,
                                          "column": col_name,
                                          "type": put.type,
                                          "parameters": put.parameters}
                    put_data = InputData(**parameters_to_pass) if put.type in LayerInputTypeChoice else OutputData(
                        **parameters_to_pass)
                    puts_data[idx][f"{idx}_{col_name}"] = put_data
            puts_data[len(puts_data) + 1] = {f"{len(puts_data) + 1}_{data.output}": OutputData(
                csv_path=data.csv_path, column=data.output, type=self.output_type, parameters=
                getattr(outputs, f"{self.output_type}Validator")(**data.dict())
            )}

        elif data_type == LayerSelectTypeChoice.folder:
            puts_data[1] = {f"1_{self.input_type.value}": InputData(
                folder_path=process_directory_paths(data.source_path),
                column=f"1_{self.input_type.value}",
                type=self.input_type,
                parameters=getattr(inputs, f"{self.input_type.value}Validator")(**data.dict())
            )}
            puts_data[2] = {f"2_{self.output_type.value}": OutputData(
                folder_path=process_directory_paths(data.target_path) if "target_path" in data.__fields_set__
                else process_directory_paths(data.source_path),
                column=f"2_{self.output_type.value}",
                type=self.output_type,
                parameters=getattr(outputs, f"{self.output_type.value}Validator")(**data.dict())
            )}

        return puts_data

    @staticmethod
    def create_put_instructions(put_data) -> Dict[int, Dict[str, Union[InputInstructionsData, OutputInstructionsData]]]:

        new_put_data = {}
        for put_id, cols_dict in put_data.items():
            new_put_data[put_id] = {}
            for col_name, put_data in cols_dict.items():
                data_to_pass = []
                if put_data.csv_path:
                    csv_table = pd.read_csv(put_data.csv_path, usecols=[put_data.column])
                    data_to_pass = csv_table.loc[:, put_data.column].tolist()
                    if put_data.type in [LayerInputTypeChoice.Image, LayerOutputTypeChoice.Segmentation]:
                        data_to_pass = [str(put_data.csv_path.parent.joinpath(elem)) for elem in data_to_pass]
                    elif put_data.type in [LayerInputTypeChoice.Categorical, LayerOutputTypeChoice.Classification]:
                        put_data.parameters.classes_names = list(set(data_to_pass))
                else:
                    for folder_path in put_data.folder_path:
                        data_to_pass.extend(
                            getattr(utils, f"extract_{put_data.type.value.lower()}_data")(folder_path,
                                                                                          put_data.parameters)
                        )

                if put_data.type == LayerInputTypeChoice.Text:
                    if put_data.parameters.pymorphy:
                        data_to_pass = apply_pymorphy(data_to_pass)

                put_type = "Input" if put_data.type in LayerInputTypeChoice else "Output"
                parameters = put_data.parameters
                if put_data.type == LayerInputTypeChoice.Categorical:
                    parameters = CategoricalValidator(**put_data.parameters.dict())
                new_put_data[put_id][col_name] = getattr(creation_data, f"{put_type}InstructionsData")(
                    type=put_data.type, parameters=parameters, data=data_to_pass
                )

        return new_put_data

    @staticmethod
    def create_table(put_instructions: Dict[int, Dict[str, Union[InputInstructionsData, OutputInstructionsData]]],
                     train_size: int,
                     shuffle: bool = True,
                     ) -> Dict[str, pd.DataFrame]:

        csv_data = {}
        for put_id, cols_dict in put_instructions.items():
            for col_name, put_data in cols_dict.items():
                csv_data[col_name] = put_data.data

        dataframe = pd.DataFrame.from_dict(csv_data)

        if shuffle:
            dataframe = dataframe.sample(frac=1)

        train_dataframe, val_dataframe = np.split(
            dataframe, [int(train_size * len(dataframe))]
        )
        dataframe = {"train": train_dataframe.reset_index(drop=True), "val": val_dataframe.reset_index(drop=True)}

        return dataframe


class CreateClassificationDataset(CreateDataset):

    def __init__(self, **kwargs):
        self.y_classes = []
        super().__init__(**kwargs)

    def create_put_instructions(self, put_data) -> \
            Dict[int, Dict[str, Union[InputInstructionsData, OutputInstructionsData]]]:

        new_put_data = {}
        for put_id, cols_dict in put_data.items():
            new_put_data[put_id] = {}
            for col_name, put_data in cols_dict.items():
                if put_data.type in LayerInputTypeChoice:
                    data_to_pass = []
                    for folder_path in put_data.folder_path:
                        data = getattr(utils, f"extract_{put_data.type.value.lower()}_data")(
                            folder_path, put_data.parameters
                        )
                        if put_id == 1:
                            self.y_classes.extend([folder_path.name for _ in data])
                        data_to_pass.extend([str(path) for path in data])
                    if put_data.type == LayerInputTypeChoice.Text:
                        if put_data.parameters.pymorphy:
                            data_to_pass = apply_pymorphy(data_to_pass)
                else:
                    data_to_pass = self.y_classes
                    put_data.parameters.classes_names = [path.name for path in put_data.folder_path]

                put_type = "Input" if put_data.type in LayerInputTypeChoice else "Output"
                new_put_data[put_id][col_name] = getattr(creation_data, f"{put_type}InstructionsData")(
                    type=put_data.type, parameters=put_data.parameters, data=data_to_pass
                )

        return new_put_data
