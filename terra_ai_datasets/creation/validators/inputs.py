from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, validator
from pydantic.types import PositiveInt, PositiveFloat

# from terra_ai_datasets.creation.validators.dataset import DataframeInputData


# --- Image validators ---
class ImageScalers(str, Enum):
    none = None
    min_max_scaler = "MinMaxScaler"
    terra_image_scaler = "TerraImageScaler"


class ImageNetworkTypes(str, Enum):
    linear = "Linear"
    convolutional = "Convolutional"


class ImageProcessTypes(str, Enum):
    stretch = "Stretch"
    fit = "Fit"
    cut = "Cut"


class ImageValidator(BaseModel):
    height: PositiveInt
    width: PositiveInt
    network: ImageNetworkTypes = ImageNetworkTypes.convolutional
    process: ImageProcessTypes = ImageProcessTypes.stretch
    preprocessing: ImageScalers = ImageScalers.none


# --- Text validators ---
class TextModeTypes(str, Enum):
    full = "Full"
    length_and_step = "Length and step"


class TextProcessTypes(str, Enum):
    embedding = "Embedding"
    bag_of_words = "Bag of words"
    word_to_vec = "Word2Vec"


class TextValidator(BaseModel):
    filters: str = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'
    preprocessing: TextProcessTypes
    max_words_count: Optional[PositiveInt]
    word2vec_size: Optional[PositiveInt]
    mode: TextModeTypes
    pymorphy: bool
    max_words: Optional[PositiveInt]
    length: Optional[PositiveInt]
    step: Optional[PositiveInt]

    @validator("mode")
    def _validate_mode(cls, value):
        if value == TextModeTypes.full:
            cls.__fields__["max_words"].required = True
            cls.__fields__["length"].required = False
            cls.__fields__["step"].required = False
        elif value == TextModeTypes.length_and_step:
            cls.__fields__["max_words"].required = False
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value

    @validator("preprocessing")
    def _validate_preprocessing(cls, value):
        if value == TextProcessTypes.word_to_vec:
            cls.__fields__["max_words_count"].required = False
            cls.__fields__["word2vec_size"].required = True
        else:
            cls.__fields__["max_words_count"].required = True
            cls.__fields__["word2vec_size"].required = False
        return value


# --- Audio validators ---
class AudioScalers(str, Enum):
    none = None
    min_max_scaler = "MinMaxScaler"
    standard_scaler = "StandardScaler"


class AudioParameterTypes(str, Enum):
    audio_signal = "Audio signal"
    chroma_stft = "Chroma stft"
    mfcc = "MFCC"
    rms = "RMS"
    spectral_centroid = "Spectral centroid"
    spectral_bandwidth = "Spectral bandwidth"
    spectral_rolloff = "Spectral rolloff"
    zero_crossing_rate = "Zero crossing rate"


class AudioResampleTypes(str, Enum):
    kaiser_best = "Kaiser best"
    kaiser_fast = "Kaiser fast"
    scipy = "Scipy"


class AudioFillTypes(str, Enum):
    last_millisecond = "Last millisecond"
    loop = "Loop"


class AudioModeTypes(str, Enum):
    full = "Full"
    length_and_step = "Length and step"


class AudioValidator(BaseModel):
    sample_rate: PositiveInt
    mode: AudioModeTypes
    parameter: List[AudioParameterTypes]
    fill_mode: AudioFillTypes
    resample: AudioResampleTypes
    preprocessing: AudioScalers
    max_seconds: Optional[PositiveFloat]
    length: Optional[PositiveFloat]
    step: Optional[PositiveFloat]

    @validator("mode")
    def _validate_mode(cls, value: AudioModeTypes) -> AudioModeTypes:
        if value == AudioModeTypes.full:
            cls.__fields__["max_seconds"].required = True
            cls.__fields__["length"].required = False
            cls.__fields__["step"].required = False
        elif value == AudioModeTypes.length_and_step:
            cls.__fields__["max_seconds"].required = False
            cls.__fields__["length"].required = True
            cls.__fields__["step"].required = True
        return value


# --- Dataframe validators ---
class CategoricalValidator(BaseModel):
    one_hot_encoding: bool
    classes_names: Optional[List[str]]


class RawValidator(BaseModel):
    pass


# --- Timeseries validators ---
class TimeseriesScalers(str, Enum):
    none = None
    min_max_scaler = "MinMaxScaler"
    standard_scaler = "StandardScaler"


class TimeseriesValidator(BaseModel):
    length: PositiveInt
    step: PositiveInt
    preprocessing: TimeseriesScalers
