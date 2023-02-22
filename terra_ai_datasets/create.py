from typing import Dict

from terra_ai_datasets.creation.dataset import CreateDataset
from terra_ai_datasets.creation.validators.tasks import LayerInputTypeChoice, LayerOutputTypeChoice, \
    LayerSelectTypeChoice


class ImageClassification(CreateDataset):
    input_type = LayerInputTypeChoice.Image
    output_type = LayerOutputTypeChoice.Classification

    def __init__(
            self,
            source_path: list,
            width: int,
            height: int,
            scaler: str,
            network: str,
            process: str,
    ):
        super().__init__(source_path=source_path, width=width, height=height, scaler=scaler, network=network,
                         process=process)


    def summary(self):
        super().summary()
        print("столько то классов и тд")


class ImageSegmentation(CreateDataset):
    input_type = LayerInputTypeChoice.Image
    output_type = LayerOutputTypeChoice.Segmentation

    def __init__(
            self,
            source_path: list,
            target_path: list,
            width: int,
            height: int,
            scaler: str,
            network: str,
            process: str,
            rgb_range: int,
            classes: Dict[str, list]
    ):
        super().__init__(source_path=source_path, target_path=target_path, width=width, height=height, scaler=scaler,
                         network=network, process=process, rgb_range=rgb_range, classes=classes)

    def preprocess_put_data(self, data, data_type: LayerSelectTypeChoice):
        inputs_data, outputs_data = super().preprocess_put_data(data, data_type)
        outputs_data[0].parameters.height = inputs_data[0].parameters.height
        outputs_data[0].parameters.width = inputs_data[0].parameters.width
        outputs_data[0].parameters.process = inputs_data[0].parameters.process
        outputs_data[0].folder_path = self.data.target_path

        return inputs_data, outputs_data

    def summary(self):
        super().summary()
        print("столько то классов и тд")


class TextClassification(CreateDataset):
    input_type = LayerInputTypeChoice.Text
    output_type = LayerOutputTypeChoice.Classification

    def summary(self):
        super().summary()
        print("столько то классов и тд")


class DataframeDataset(CreateDataset):
    input_type = LayerInputTypeChoice.Dataframe
    output_type = "Dataset"

    def summary(self):
        super().summary()
        print("столько то классов и тд")
