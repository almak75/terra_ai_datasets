from typing import Dict, Union, Any

from terra_ai_datasets.creation.dataset import CreateDataset, CreateClassificationDataset
from terra_ai_datasets.creation.validators.creation_data import InputData, OutputData
from terra_ai_datasets.creation.validators import inputs as val_inputs
from terra_ai_datasets.creation.validators import outputs as val_outputs
from terra_ai_datasets.creation.validators.inputs import ImageScalers, TextModeTypes, TextProcessTypes
from terra_ai_datasets.creation.validators.tasks import LayerInputTypeChoice, LayerOutputTypeChoice, \
    LayerSelectTypeChoice


class ImageClassification(CreateClassificationDataset):
    input_type = LayerInputTypeChoice.Image
    output_type = LayerOutputTypeChoice.Classification

    def __init__(
            self,
            source_path: list,
            train_size: float,
            width: int,
            height: int,
            network: str,
            process: str,
            preprocessing: str = ImageScalers.none,
            one_hot_encoding: bool = True
    ):
        super().__init__(source_path=source_path, train_size=train_size, width=width, height=height, network=network,
                         process=process, preprocessing=preprocessing, one_hot_encoding=one_hot_encoding)


class ImageSegmentation(CreateDataset):
    input_type = LayerInputTypeChoice.Image
    output_type = LayerOutputTypeChoice.Segmentation

    def __init__(
            self,
            source_path: list,
            target_path: list,
            train_size: float,
            width: int,
            height: int,
            network: str,
            process: str,
            rgb_range: int,
            classes: Dict[str, list],
            preprocessing: str = ImageScalers.none
    ):
        super().__init__(source_path=source_path, target_path=target_path, train_size=train_size,  width=width,
                         height=height, preprocessing=preprocessing, network=network, process=process,
                         rgb_range=rgb_range, classes=classes)

    def preprocess_put_data(self, data, data_type: LayerSelectTypeChoice):
        puts_data = super().preprocess_put_data(data, data_type)

        puts_data[2][f"2_{self.output_type.value}"].parameters.height = \
            puts_data[1][f"1_{self.input_type.value}"].parameters.height
        puts_data[2][f"2_{self.output_type.value}"].parameters.width = \
            puts_data[1][f"1_{self.input_type.value}"].parameters.width
        puts_data[2][f"2_{self.output_type.value}"].parameters.process = \
            puts_data[1][f"1_{self.input_type.value}"].parameters.process

        return puts_data

    def summary(self):
        super().summary()
        text_to_print = f"\n\033[1mКлассы в масках сегментации и их цвета в RGB:\033[0m"
        classes = self.put_instructions[2][f"2_{self.output_type.value}"].parameters.classes
        for name, color in classes.items():
            text_to_print += f"\n{name}: {color.as_rgb_tuple()}"
        print(text_to_print)


class TextClassification(CreateClassificationDataset):
    input_type = LayerInputTypeChoice.Text
    output_type = LayerOutputTypeChoice.Classification

    def __init__(
            self,
            source_path: list,
            train_size: float,
            preprocessing: str = TextProcessTypes.embedding,
            max_words_count: int = 20000,
            mode: str = TextModeTypes.full,
            filters: str = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff',
            max_words: int = None,
            length: int = None,
            step: int = None,
            pymorphy: bool = False,
            one_hot_encoding: bool = True
    ):
        parameters = {"source_path": source_path, "train_size": train_size, "preprocessing": preprocessing,
                      "max_words_count": max_words_count, "mode": mode, "filters": filters, "pymorphy": pymorphy,
                      "one_hot_encoding": one_hot_encoding}

        for name, param in {"max_words": max_words, "length": length, "step": step}.items():
            if param:
                parameters[name] = param

        super().__init__(**parameters)


class AudioClassification(CreateClassificationDataset):
    input_type = LayerInputTypeChoice.Audio
    output_type = LayerOutputTypeChoice.Classification

    def __init__(
            self,
            source_path: list,
            train_size: float,
            mode: str,
            parameter: list,
            fill_mode: str,
            resample: str,
            max_seconds: float = None,
            length: float = None,
            step: float = None,
            sample_rate: int = 22050,
            preprocessing: str = 'None',
            one_hot_encoding: bool = True
    ):
        parameters = {"source_path": source_path, "train_size": train_size, "preprocessing": preprocessing,
                      "parameter": parameter, "mode": mode, "sample_rate": sample_rate, "fill_mode": fill_mode,
                      "resample": resample, "one_hot_encoding": one_hot_encoding}

        for name, param in {"max_seconds": max_seconds, "length": length, "step": step}.items():
            if param:
                parameters[name] = param

        super().__init__(**parameters)

    def preprocess_put_data(self, data, data_type: LayerSelectTypeChoice):
        put_data = super().preprocess_put_data(data, data_type)

        input_data = put_data[1][f"1_{self.input_type.value}"]
        new_put_data = {}

        for put_id, parameter in enumerate(input_data.parameters.parameter, 1):
            new_params_data = input_data.parameters.dict()
            new_params_data['parameter'] = [parameter]
            new_input_data = InputData(
                folder_path=input_data.folder_path,
                type=self.input_type,
                parameters=getattr(val_inputs, f"{self.input_type.value}Validator")(**new_params_data)
            )
            new_put_data[put_id] = {f"{put_id}_Audio": new_input_data}

        new_put_data[len(new_put_data) + 1] = \
            {f"{len(new_put_data) + 1}_{self.output_type.value}": put_data[2][f"2_{self.output_type.value}"]}

        return new_put_data


class Dataframe(CreateDataset):

    def __init__(
            self,
            csv_path: str,
            train_size: float,
            inputs: list,
            outputs: list,
            **kwargs
    ):
        super().__init__(csv_path=csv_path, inputs=inputs, outputs=outputs,  train_size=train_size, **kwargs)


class DataframeRegression(Dataframe):
    input_type = LayerInputTypeChoice.Dataframe
    output_type = LayerOutputTypeChoice.Regression


class TimeseriesDepth(Dataframe):
    input_type = LayerInputTypeChoice.Timeseries
    output_type = LayerOutputTypeChoice.Depth

    def __init__(
            self,
            csv_path: str,
            train_size: float,
            inputs: list,
            outputs: list,
            preprocessing: str,
            length: int,
            step: int,
            depth: int,
    ):
        super().__init__(csv_path=csv_path, inputs=inputs, outputs=outputs,  train_size=train_size,
                         preprocessing=preprocessing, length=length, step=step, depth=depth)

    def preprocess_put_data(self, data, data_type: LayerSelectTypeChoice) -> \
            Dict[int, Union[Dict[Any, Any], Dict[str, InputData], Dict[str, OutputData]]]:

        puts_data = {1: {}, 2: {}}
        for idx, col_name in enumerate(self.data.inputs, 1):
            parameters_to_pass = {"csv_path": self.data.csv_path,
                                  "column": col_name,
                                  "type": self.input_type,
                                  "parameters": getattr(val_inputs, f"{self.input_type.value}Validator")(
                                      **self.data.dict())}
            put_data = InputData(**parameters_to_pass)
            puts_data[1][f"{idx}_{col_name}"] = put_data
        for idx, col_name in enumerate(self.data.outputs, 1 + len(self.data.inputs)):
            parameters_to_pass = {"csv_path": self.data.csv_path,
                                  "column": col_name,
                                  "type": self.output_type,
                                  "parameters": getattr(val_outputs, f"{self.output_type.value}Validator")(
                                      **self.data.dict())}
            put_data = OutputData(**parameters_to_pass)
            puts_data[2][f"{idx}_{col_name}"] = put_data

        return puts_data


class TimeseriesTrend(Dataframe):
    input_type = LayerInputTypeChoice.Timeseries
    output_type = LayerOutputTypeChoice.Trend

    def __init__(
            self,
            csv_path: str,
            train_size: float,
            inputs: list,
            outputs: list,
            preprocessing: str,
            length: int,
            step: int,
            deviation: float,
            one_hot_encoding: bool
    ):
        super().__init__(csv_path=csv_path, inputs=inputs, outputs=outputs,  train_size=train_size,
                         preprocessing=preprocessing, length=length, step=step, deviation=deviation,
                         one_hot_encoding=one_hot_encoding)

    def preprocess_put_data(self, data, data_type: LayerSelectTypeChoice) -> \
            Dict[int, Union[Dict[Any, Any], Dict[str, InputData], Dict[str, OutputData]]]:

        puts_data = {1: {}, 2: {}}
        for idx, col_name in enumerate(self.data.inputs, 1):
            parameters_to_pass = {"csv_path": self.data.csv_path,
                                  "column": col_name,
                                  "type": self.input_type,
                                  "parameters": getattr(val_inputs, f"{self.input_type.value}Validator")(
                                      **self.data.dict())}
            put_data = InputData(**parameters_to_pass)
            puts_data[1][f"{idx}_{col_name}"] = put_data
        for idx, col_name in enumerate(self.data.outputs, 1 + len(self.data.inputs)):
            parameters_to_pass = {"csv_path": self.data.csv_path,
                                  "column": col_name,
                                  "type": self.output_type,
                                  "parameters": getattr(val_outputs, f"{self.output_type.value}Validator")(
                                      **self.data.dict())}
            put_data = OutputData(**parameters_to_pass)
            puts_data[2][f"{idx}_{col_name}"] = put_data

        return puts_data
