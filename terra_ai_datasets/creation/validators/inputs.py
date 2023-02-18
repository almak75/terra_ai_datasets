from enum import Enum
from pydantic import BaseModel, validator
from pydantic.types import PositiveInt


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
    image_scaler: ImageScalers = ImageScalers.none


# --- Text validators ---
class TextFormatTypes(str, Enum):
    full = "Full"
    length_and_step = "Length and step"


class TextProcessTypes(str, Enum):
    embedding = "Embedding"
    bag_of_words = "Bag of words"
    word_to_vec = "Word2Vec"


class TextValidator(BaseModel):
    filters: str = '–—!"#$%&()*+,-./:;<=>?@[\\]^«»№_`{|}~\t\n\xa0–\ufeff'
    max_words_count: PositiveInt
    text_format: TextFormatTypes
    pymorphy: bool
