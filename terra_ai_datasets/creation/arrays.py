from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer

from terra_ai_datasets.creation.utils import resize_frame
from terra_ai_datasets.creation.validators.inputs import ImageNetworkTypes, ImageValidator, TextValidator, \
    TextProcessTypes, TextModeTypes
from terra_ai_datasets.creation.validators.outputs import SegmentationValidator, ClassificationValidator


class Array(ABC):

    @abstractmethod
    def create(self, source: Any, parameters: Any):
        pass

    @abstractmethod
    def preprocess(self, array: np.ndarray, preprocess: Any, parameters: Any):
        pass


class ImageArray(Array):

    def create(self, source: str, parameters: ImageValidator):

        image = Image.open(source)
        array = np.asarray(image)
        array = resize_frame(image_array=array,
                             target_shape=(parameters.height, parameters.width),
                             frame_mode=parameters.process)
        if parameters.network == ImageNetworkTypes.linear:
            array = array.reshape(np.prod(np.array(array.shape)))

        return array

    def preprocess(self, array: np.ndarray, preprocess_obj: Any, parameters: ImageValidator) -> np.ndarray:
        orig_shape = array.shape
        array = preprocess_obj.transform(array.reshape(-1, 1))
        array = array.reshape(orig_shape)

        return array


class TextArray(Array):

    def create(self, source: str, parameters: TextValidator):

        return source

    def preprocess(self, text_list: list, preprocess_obj: Tokenizer, parameters: TextValidator) -> np.ndarray:

        array = []
        for text in text_list:
            if parameters.preprocessing == TextProcessTypes.embedding:
                text_array = preprocess_obj.texts_to_sequences([text])[0]
                if parameters.mode == TextModeTypes.full and len(text_array) < parameters.max_words:
                    text_array += [0 for _ in range(parameters.max_words - len(text_array))]
                elif parameters.mode == TextModeTypes.length_and_step and len(text_array) < parameters.length:
                    text_array += [0 for _ in range(parameters.length - len(text_array))]
            elif parameters.preprocessing == TextProcessTypes.bag_of_words:
                text_array = preprocess_obj.texts_to_matrix([text])[0]
            array.append(text_array)

        return np.array(array)


class ClassificationArray(Array):

    def create(self, source: str, parameters: ClassificationValidator):

        array = parameters.classes_names.index(source)
        if parameters.one_hot_encoding:
            zeros = np.zeros(len(parameters.classes_names))
            zeros[array] = 1
            array = zeros

        return array

    def preprocess(self, array: np.ndarray, preprocess_obj, parameters: SegmentationValidator):

        return array


class SegmentationArray(Array):

    def create(self, source: str, parameters: SegmentationValidator):

        image = Image.open(source)
        array = np.asarray(image)
        array = resize_frame(image_array=array,
                             target_shape=(parameters.height, parameters.width),
                             frame_mode=parameters.process)

        array = self.image_to_ohe(array, parameters)

        return array

    def preprocess(self, array: np.ndarray, preprocess_obj, parameters: SegmentationValidator):

        return array

    @staticmethod
    def image_to_ohe(img_array, parameters: SegmentationValidator):
        mask_ohe = []
        mask_range = parameters.rgb_range
        for color_obj in parameters.classes.values():
            color = color_obj.as_rgb_tuple()
            color_array = np.expand_dims(np.where((color[0] + mask_range >= img_array[:, :, 0]) &
                                                  (img_array[:, :, 0] >= color[0] - mask_range) &
                                                  (color[1] + mask_range >= img_array[:, :, 1]) &
                                                  (img_array[:, :, 1] >= color[1] - mask_range) &
                                                  (color[2] + mask_range >= img_array[:, :, 2]) &
                                                  (img_array[:, :, 2] >= color[2] - mask_range), 1, 0),
                                         axis=2)
            mask_ohe.append(color_array)

        return np.concatenate(np.array(mask_ohe), axis=2).astype(np.uint8)
