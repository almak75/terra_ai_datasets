from enum import Enum


class TasksChoice(str, Enum):
    ImageClassification = "ImageClassification"
    ImageSegmentation = "ImageSegmentation"
    ImageRegression = "ImageRegression"
    TextClassification = "TextClassification"
    TextSegmentation = "TextSegmentation"
    TextRegression = "TextRegression"
    AudioClassification = "AudioClassification"
    DataframeClassification = "DataframeClassification"
    DataframeRegression = "DataframeRegression"
    DataframeTimeseries = "DataframeTimeseries"


class LayerInputTypeChoice(str, Enum):
    Image = "Image"
    Text = "Text"
    Audio = "Audio"
    Dataframe = "Dataframe"
    # Scaler = "Scaler"
    # Raw = "Raw"


class LayerOutputTypeChoice(str, Enum):
    Classification = "Classification"
    Segmentation = "Segmentation"
    Regression = "Regression"
    Timeseries = "Timeseries"
    TimeseriesTrend = "TimeseriesTrend"


class LayerSelectTypeChoice(str, Enum):
    table = "table"
    folder = "folder"
