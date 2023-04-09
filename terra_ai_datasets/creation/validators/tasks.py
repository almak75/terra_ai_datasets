from enum import Enum


class TasksChoice(str, Enum):
    ImageClassification = "ImageClassification"
    ImageSegmentation = "ImageSegmentation"
    TextClassification = "TextClassification"
    AudioClassification = "AudioClassification"
    DataframeClassification = "DataframeClassification"
    DataframeRegression = "DataframeRegression"
    TimeseriesDepth = "TimeseriesDepth"
    TimeseriesTrend = "TimeseriesTrend"


class LayerInputTypeChoice(str, Enum):
    Image = "Image"
    Text = "Text"
    Audio = "Audio"
    Dataframe = "Dataframe"
    Categorical = "Categorical"
    Raw = "Raw"
    Timeseries = "Timeseries"
    # Scaler = "Scaler"


class LayerOutputTypeChoice(str, Enum):
    Classification = "Classification"
    Segmentation = "Segmentation"
    Regression = "Regression"
    Dataset = "Dataset"
    Trend = "Trend"
    Depth = "Depth"


class LayerSelectTypeChoice(str, Enum):
    table = "table"
    folder = "folder"
