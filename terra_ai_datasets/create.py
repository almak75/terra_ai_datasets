from terra_ai_datasets.creation.dataset import CreateDataset
from terra_ai_datasets.creation.validators.tasks import LayerInputTypeChoice, LayerOutputTypeChoice


class ImageClassification(CreateDataset):
    input_type = LayerInputTypeChoice.Image
    output_type = LayerOutputTypeChoice.Classification

    def summary(self):
        super().summary()
        print("столько то классов и тд")


class ImageSegmentation(CreateDataset):
    input_type = LayerInputTypeChoice.Image
    output_type = LayerOutputTypeChoice.Segmentation

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
