import os.path
from typing import Dict, Union, Any

from terra_ai_datasets.creation.dataset import CreateDataset, CreateClassificationDataset
from terra_ai_datasets.creation.utils import get_classes_annotation, get_classes_autosearch
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
        """
        Класс подготовки датасета для задачи классификации изображений.

        :param source_path: Список относительных путей до папок с изображениями;
        :param train_size: Соотношение обучающей выборки к валидационной;
        :param width: Ширина изображений;
        :param height: Высота изображений;
        :param process: Метод обработки изображений при изменении размерности. Варианты: "Stretch", "Fit", "Cut";
        :param network: Постобработка массивов под определенный вид нейронной сети. Варианты: "Convolutional", "Linear";
        :param preprocessing: Выбор скейлера. Варианты: "None", "MinMaxScaler", "TerraImageScaler";
        :param one_hot_encoding: Перевод Y массивов в формат One-Hot Encoding.
        """
        super().__init__(source_path=source_path, train_size=train_size, width=width, height=height, process=process,
                         network=network, preprocessing=preprocessing, one_hot_encoding=one_hot_encoding)


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
            classes: Dict[str, list] = None,
            num_classes: int = None,
            classes_path: str = None,
            preprocessing: str = ImageScalers.none
    ):
        """
        Класс подготовки датасета для задачи сегментации изображений.

        :param source_path: Список относительных путей до папок с изображениями;
        :param target_path: Список относительных путей до папок с масками сегментации;
        :param train_size: Соотношение обучающей выборки к валидационной;
        :param width: Ширина изображений;
        :param height: Высота изображений;
        :param process: Метод обработки изображений при изменении размерности. Варианты: "Stretch", "Fit", "Cut";
        :param network: Постобработка массивов под определенный вид нейронной сети. Варианты: "Convolutional", "Linear";
        :param preprocessing: Выбор скейлера. Варианты: "None", "MinMaxScaler", "TerraImageScaler";
        :param rgb_range: Диапазон, при котором пиксели будут отнесены к классу;
        :param classes: Названия классов и их RGB значения в виде словаря;
        :param classes_path: Путь к txt файлу с указанием названия классов и их RGB значений.
        """
        if not classes and not classes_path and num_classes:
            classes = get_classes_autosearch(source=target_path, num_classes=num_classes, mask_range=rgb_range,
                                             height=height, width=width, frame_mode=process)
        elif not classes and classes_path:
            classes = get_classes_annotation(classes_path)

        super().__init__(source_path=source_path, target_path=target_path, train_size=train_size,  width=width,
                         height=height, process=process, network=network, preprocessing=preprocessing,
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


class TextClassification(CreateClassificationDataset):
    input_type = LayerInputTypeChoice.Text
    output_type = LayerOutputTypeChoice.Classification

    def __init__(
            self,
            source_path: list,
            train_size: float,
            preprocessing: str = TextProcessTypes.embedding,
            max_words_count: int = None,
            word2vec_size: int = None,
            mode: str = TextModeTypes.full,
            filters: str = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff',
            max_words: int = None,
            length: int = None,
            step: int = None,
            pymorphy: bool = False,
            one_hot_encoding: bool = True
    ):
        """
        Класс подготовки датасета для задачи классификации текстов.

        :param source_path: Список относительных путей до папок с изображениями;
        :param train_size: Соотношение обучающей выборки к валидационной;
        :param preprocessing: Тип обработки текстов. Варианты: "Embedding", "Bag of words", "Word2Vec";
        :param max_words_count: Максимальное количество слов, сохраняемое в словаре частотности;
        :param mode: Режим обрезки текстовых файлов. Варианты: "Full", "Length and step";
        :param filters: Символы, подлежащие удалению;
        :param pymorphy: Применить морфологический анализатор (приведение слов в нормальную форму);
        :param max_words: ТОЛЬКО при mode == "Full" - Максимальное количество слов, извлекаемое из одного файла;
        :param length: ТОЛЬКО при mode == "Length and step" - Количество слов в одном примере;
        :param step: ТОЛЬКО при mode == "Length and step" - Шаг движения по тексту при составлении примеров;
        :param one_hot_encoding: Перевод Y массивов в формат One-Hot Encoding.
        """
        parameters = {"source_path": source_path, "train_size": train_size, "preprocessing": preprocessing,
                      "max_words_count": max_words_count, "mode": mode, "filters": filters, "pymorphy": pymorphy,
                      "one_hot_encoding": one_hot_encoding}

        for name, param in {"max_words": max_words, "length": length, "step": step,
                            "word2vec_size": word2vec_size}.items():
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
        """
        Класс подготовки датасета для задачи классификации аудио.

        :param source_path: Список относительных путей до папок с изображениями;
        :param train_size: Соотношение обучающей выборки к валидационной;
        :param mode: Режим обрезки аудио файлов. Варианты: "Full", "Length and step";
        :param parameter: Спсиок извлекаемых фичей из аудио файлов;
        :param fill_mode: Режим заполнения при недостатке длины. Варианты: "Loop", "Last millisecond";
        :param resample: Режим ресемпла при открытии аудио файла. Варианты: "Kaiser best", "Kaiser fast", "Scipy";
        :param max_seconds: ТОЛЬКО при mode == "Full" - Максимальное количество секунд, извлекаемое из одного файла;
        :param length: ТОЛЬКО при mode == "Length and step" - Длина одного примера;
        :param step: ТОЛЬКО при mode == "Length and step" - Шаг движения по аудиофайлу при составлении примеров;
        :param sample_rate: Sample rate при открытии аудиофайла;
        :param preprocessing: Выбор скейлера. Варианты: "StandardScaler", "MinMaxScaler";
        :param one_hot_encoding: Перевод Y массивов в формат One-Hot Encoding.
        """
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
            **kwargs
    ):
        super().__init__(**kwargs)


class DataframeRegression(Dataframe):
    input_type = LayerInputTypeChoice.Dataframe
    output_type = LayerOutputTypeChoice.Regression

    def __init__(
            self,
            csv_path: str,
            train_size: float,
            inputs: list,
            output: str,
            preprocessing: str
    ):
        super().__init__(csv_path=csv_path, inputs=inputs, output=output,  train_size=train_size,
                         preprocessing=preprocessing)


class DataframeClassification(Dataframe):
    input_type = LayerInputTypeChoice.Dataframe
    output_type = LayerOutputTypeChoice.Classification

    def __init__(
            self,
            csv_path: str,
            train_size: float,
            inputs: list,
            output: str,
            one_hot_encoding: bool,
    ):
        super().__init__(csv_path=csv_path, inputs=inputs, output=output,  train_size=train_size,
                         one_hot_encoding=one_hot_encoding)


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
        """
        Класс подготовки датасета для задачи временных рядов.

        :param csv_path: Путь до csv-файла;
        :param train_size: Соотношение обучающей выборки к валидационной;
        :param inputs: Список колонок, используемых в качестве входа;
        :param outputs: Список колонок, используемых в качестве выхода;
        :param preprocessing: Выбор скейлера. Варианты: "StandardScaler", "MinMaxScaler";
        :param length: Длина одного примера;
        :param step: Длина шага окна разбивки при составлении примеров;
        :param depth: Глубина предсказания;
        """
        super().__init__(csv_path=csv_path, inputs=inputs, outputs=outputs,  train_size=train_size,
                         preprocessing=preprocessing, length=length, step=step, depth=depth)

    def preprocess_put_data(self, data, data_type: LayerSelectTypeChoice) -> \
            Dict[int, Union[Dict[Any, Any], Dict[str, InputData], Dict[str, OutputData]]]:

        puts_data = {1: {}, 2: {}}
        for idx, col_name in enumerate(data.inputs, 1):
            parameters_to_pass = {"csv_path": data.csv_path,
                                  "column": col_name,
                                  "type": self.input_type,
                                  "parameters": getattr(val_inputs, f"{self.input_type.value}Validator")(
                                      **data.dict())}
            put_data = InputData(**parameters_to_pass)
            puts_data[1][f"{idx}_{col_name}"] = put_data
        for idx, col_name in enumerate(data.outputs, 1 + len(data.inputs)):
            parameters_to_pass = {"csv_path": data.csv_path,
                                  "column": col_name,
                                  "type": self.output_type,
                                  "parameters": getattr(val_outputs, f"{self.output_type.value}Validator")(
                                      **data.dict())}
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
            output: str,
            preprocessing: str,
            length: int,
            step: int,
            deviation: float,
            one_hot_encoding: bool
    ):
        """
        Класс подготовки датасета для задачи временных рядов с предсказанием тренда.

        :param csv_path: Путь до csv-файла;
        :param train_size: Соотношение обучающей выборки к валидационной;
        :param inputs: Список колонок, используемых в качестве входа;
        :param outputs: Колонка, используемая в качестве выхода;
        :param preprocessing: Выбор скейлера. Варианты: "StandardScaler", "MinMaxScaler";
        :param length: Длина одного примера;
        :param step: Шаг движения по аудиофайлу при составлении примеров;
        :param deviation: Отклонение нулевого тренда в процентах;
        :param one_hot_encoding: Перевод Y массивов в формат One-Hot Encoding.
        """
        super().__init__(csv_path=csv_path, inputs=inputs, output=output,  train_size=train_size,
                         preprocessing=preprocessing, length=length, step=step, deviation=deviation,
                         one_hot_encoding=one_hot_encoding)

    def preprocess_put_data(self, data, data_type: LayerSelectTypeChoice) -> \
            Dict[int, Union[Dict[Any, Any], Dict[str, InputData], Dict[str, OutputData]]]:

        puts_data = {1: {}, 2: {}}
        for idx, col_name in enumerate(data.inputs, 1):
            parameters_to_pass = {"csv_path": data.csv_path,
                                  "column": col_name,
                                  "type": self.input_type,
                                  "parameters": getattr(val_inputs, f"{self.input_type.value}Validator")(
                                      **data.dict())}
            put_data = InputData(**parameters_to_pass)
            puts_data[1][f"{idx}_{col_name}"] = put_data
        parameters_to_pass = {"csv_path": data.csv_path,
                              "column": data.output,
                              "type": self.output_type,
                              "parameters": getattr(val_outputs, f"{self.output_type.value}Validator")(
                                  **data.dict())}
        put_data = OutputData(**parameters_to_pass)
        puts_data[2][f"{len(puts_data[1]) + 1}_{data.output}"] = put_data

        return puts_data
