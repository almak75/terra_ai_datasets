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


class LayerEncodingChoice(str, Enum):
    none = "none"
    ohe = "ohe"
    multi = "multi"


class LayerDatatypeChoice(str, Enum):
    dim = "DIM"
    one_dim = "1D"
    two_dim = "2D"
    three_dim = "3D"


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
