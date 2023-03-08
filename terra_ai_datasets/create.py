from typing import Dict

from terra_ai_datasets.creation.dataset import CreateDataset, CreateClassificationDataset
from terra_ai_datasets.creation.validators.creation_data import InputData
from terra_ai_datasets.creation.validators import inputs
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
        inputs_data, outputs_data = super().preprocess_put_data(data, data_type)
        outputs_data[0].parameters.height = inputs_data[0].parameters.height
        outputs_data[0].parameters.width = inputs_data[0].parameters.width
        outputs_data[0].parameters.process = inputs_data[0].parameters.process
        outputs_data[0].folder_path = data.target_path

        return inputs_data, outputs_data

    def summary(self):
        super().summary()
        print("столько то классов и тд")


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
        _, outputs_data = super().preprocess_put_data(data, data_type)

        inputs_data = []
        for parameter in data.parameter:
            params = data.dict()
            params['parameter'] = [parameter]
            inputs_data.append(InputData(
                folder_path=data.source_path,
                type=self.input_type,
                parameters=getattr(inputs, f"{self.input_type.value}Validator")(**params)
            ))

        return inputs_data, outputs_data


class DataframeDataset(CreateDataset):
    input_type = LayerInputTypeChoice.Dataframe
    output_type = "Dataset"
