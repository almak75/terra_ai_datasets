from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np
import librosa.feature as librosa_feature
from librosa import load as librosa_load
from PIL import Image
from tensorflow.keras.preprocessing.text import Tokenizer

from terra_ai_datasets.creation.validators.inputs import ImageNetworkTypes, ImageValidator, TextValidator, \
    TextProcessTypes, TextModeTypes, AudioValidator, AudioParameterTypes, RawValidator, TimeseriesValidator, \
    ImageScalers, ImageProcessTypes
from terra_ai_datasets.creation.validators.outputs import SegmentationValidator, ClassificationValidator, \
    RegressionValidator, DepthValidator, TrendValidator


def resize_frame(image_array, target_shape, frame_mode):
    original_shape = (image_array.shape[0], image_array.shape[1])
    resized = None
    if frame_mode == ImageProcessTypes.stretch:
        resized = cv2.resize(image_array, (target_shape[1], target_shape[0]))

    elif frame_mode == ImageProcessTypes.fit:
        if image_array.shape[1] >= image_array.shape[0]:
            resized_shape = list(target_shape).copy()
            resized_shape[0] = int(
                image_array.shape[0] / (image_array.shape[1] / target_shape[1])
            )
            if resized_shape[0] > target_shape[0]:
                resized_shape = list(target_shape).copy()
                resized_shape[1] = int(
                    image_array.shape[1] / (image_array.shape[0] / target_shape[0])
                )
            image_array = cv2.resize(image_array, (resized_shape[1], resized_shape[0]))
        elif image_array.shape[0] >= image_array.shape[1]:
            resized_shape = list(target_shape).copy()
            resized_shape[1] = int(
                image_array.shape[1] / (image_array.shape[0] / target_shape[0])
            )
            if resized_shape[1] > target_shape[1]:
                resized_shape = list(target_shape).copy()
                resized_shape[0] = int(
                    image_array.shape[0] / (image_array.shape[1] / target_shape[1])
                )
            image_array = cv2.resize(image_array, (resized_shape[1], resized_shape[0]))
        resized = image_array
        if resized.shape[0] < target_shape[0]:
            black_bar = np.zeros(
                (int((target_shape[0] - resized.shape[0]) / 2), resized.shape[1], 3),
                dtype="uint8",
            )
            resized = np.concatenate((black_bar, resized))
            black_bar_2 = np.zeros(
                (int((target_shape[0] - resized.shape[0])), resized.shape[1], 3),
                dtype="uint8",
            )
            resized = np.concatenate((resized, black_bar_2))
        if resized.shape[1] < target_shape[1]:
            black_bar = np.zeros(
                (target_shape[0], int((target_shape[1] - resized.shape[1]) / 2), 3),
                dtype="uint8",
            )
            resized = np.concatenate((black_bar, resized), axis=1)
            black_bar_2 = np.zeros(
                (target_shape[0], int((target_shape[1] - resized.shape[1])), 3),
                dtype="uint8",
            )
            resized = np.concatenate((resized, black_bar_2), axis=1)

    elif frame_mode == ImageProcessTypes.cut:
        resized = image_array.copy()
        if original_shape[0] > target_shape[0]:
            resized = resized[
                int(original_shape[0] / 2 - target_shape[0] / 2) : int(
                    original_shape[0] / 2 - target_shape[0] / 2
                )
                + target_shape[0],
                :,
            ]
        else:
            black_bar = np.zeros(
                (int((target_shape[0] - original_shape[0]) / 2), original_shape[1], 3),
                dtype="uint8",
            )
            resized = np.concatenate((black_bar, resized))
            black_bar_2 = np.zeros(
                (int((target_shape[0] - resized.shape[0])), original_shape[1], 3),
                dtype="uint8",
            )
            resized = np.concatenate((resized, black_bar_2))
        if original_shape[1] > target_shape[1]:
            resized = resized[
                :,
                int(original_shape[1] / 2 - target_shape[1] / 2) : int(
                    original_shape[1] / 2 - target_shape[1] / 2
                )
                + target_shape[1],
            ]
        else:
            black_bar = np.zeros(
                (target_shape[0], int((target_shape[1] - original_shape[1]) / 2), 3),
                dtype="uint8",
            )
            resized = np.concatenate((black_bar, resized), axis=1)
            black_bar_2 = np.zeros(
                (target_shape[0], int((target_shape[1] - resized.shape[1])), 3),
                dtype="uint8",
            )
            resized = np.concatenate((resized, black_bar_2), axis=1)
    return resized


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
        if parameters.preprocessing == ImageScalers.terra_image_scaler:
            array = array.astype(np.float32)
            if len(array.shape) > 3:
                for i in range(len(array)):
                    array[i] = preprocess_obj.transform(array[i])
            else:
                array = preprocess_obj.transform(array)
        else:
            orig_shape = array.shape
            array = preprocess_obj.transform(array.reshape(-1, 1)).astype(np.float32)
            array = array.reshape(orig_shape)

        return array


class TextArray(Array):

    def create(self, source: str, parameters: TextValidator):

        return source

    def preprocess(self, text_list: list, preprocess_obj: Tokenizer, parameters: TextValidator) -> np.ndarray:

        array = []
        if not type(text_list) == np.ndarray:
            text_list = [text_list]
        for text in text_list:
            if parameters.preprocessing == TextProcessTypes.embedding:
                text_array = preprocess_obj.texts_to_sequences([text])[0]
                if parameters.mode == TextModeTypes.full and len(text_array) < parameters.max_words:
                    text_array += [0 for _ in range(parameters.max_words - len(text_array))]
                elif parameters.mode == TextModeTypes.full and len(text_array) > parameters.max_words:
                    text_array = text_array[:parameters.max_words]
                elif parameters.mode == TextModeTypes.length_and_step and len(text_array) < parameters.length:
                    text_array += [0 for _ in range(parameters.length - len(text_array))]
            elif parameters.preprocessing == TextProcessTypes.bag_of_words:
                text_array = preprocess_obj.texts_to_matrix([text])[0].astype(np.int8)
            elif parameters.preprocessing == TextProcessTypes.word_to_vec:
                text_array = []
                for word in text.split(' '):
                    try:
                        text_array.append(preprocess_obj.wv[word])
                    except KeyError:
                        text_array.append(np.zeros((parameters.word2vec_size,)))
                if len(text_array) < parameters.length:
                    words_to_add = [[0 for _ in range(parameters.word2vec_size)]
                                    for _ in range(parameters.length - len(text_array))]
                    text_array += words_to_add
            array.append(text_array)

        return np.array(array)


class AudioArray(Array):

    def create(self, source: str, parameters: AudioValidator):
        array = []
        parameter = parameters.parameter[0]
        audio_path, start_stop = source.split(';')
        offset, stop = [float(x) for x in start_stop.split(':')]
        duration = stop - offset

        y, sr = librosa_load(
            path=audio_path, sr=parameters.sample_rate, offset=offset, duration=duration,
            res_type=parameters.resample.name
        )

        if round(parameters.sample_rate * duration, 0) > y.shape[0]:
            zeros = np.zeros((int(round(parameters.sample_rate * duration, 0)) - y.shape[0],))
            y = np.concatenate((y, zeros))
        if parameter in [AudioParameterTypes.chroma_stft, AudioParameterTypes.mfcc,
                         AudioParameterTypes.spectral_centroid, AudioParameterTypes.spectral_bandwidth,
                         AudioParameterTypes.spectral_rolloff]:
            array = getattr(librosa_feature, parameter.name)(y=y, sr=parameters.sample_rate)
        elif parameter == AudioParameterTypes.rms:
            array = getattr(librosa_feature, parameter.name)(y=y)[0]
        elif parameter == AudioParameterTypes.zero_crossing_rate:
            array = getattr(librosa_feature, parameter.name)(y=y)
        elif parameter == AudioParameterTypes.audio_signal:
            array = y

        array = np.array(array)
        if len(array.shape) == 2:
            array = array.transpose()
        if array.dtype == 'float64':
            array = array.astype('float32')

        return array

    def preprocess(self, array: np.ndarray, preprocess: Any, parameters: Any):

        orig_shape = array.shape
        if len(orig_shape) > 1:
            array = array.reshape(-1, 1)
        array = preprocess.transform(array)
        array = array.reshape(orig_shape)

        return array


class RawArray(Array):

    def create(self, source: str, parameters: RawValidator):

        array = np.array([source])

        return array

    def preprocess(self, array: np.ndarray, preprocess: Any, parameters: RawValidator):

        return array


class TimeseriesArray(Array):

    def create(self, source: list, parameters: TimeseriesValidator):

        array = np.array(source)
        array = np.expand_dims(array, 1)

        return array

    def preprocess(self, array: np.ndarray, preprocess: Any, parameters: TimeseriesValidator):

        orig_shape = array.shape
        array = array.reshape(-1, 1)
        array = preprocess.transform(array)
        array = array.reshape(orig_shape)

        return array


class ClassificationArray(Array):

    def create(self, source: str, parameters: ClassificationValidator):

        array = np.array([parameters.classes_names.index(source)], dtype=np.uint8)
        if parameters.one_hot_encoding:
            zeros = np.zeros(len(parameters.classes_names), dtype=np.uint8)
            zeros[array[0]] = 1
            array = zeros

        return array

    def preprocess(self, array: np.ndarray, preprocess: Any, parameters: ClassificationValidator):

        return array


class CategoricalArray(ClassificationArray):
    pass


class SegmentationArray(Array):

    def create(self, source: str, parameters: SegmentationValidator):

        image = Image.open(source)
        array = np.asarray(image)
        array = resize_frame(image_array=array,
                             target_shape=(parameters.height, parameters.width),
                             frame_mode=parameters.process)
        array = self.image_to_ohe(array, parameters)

        return array

    def preprocess(self, array: np.ndarray, preprocess: Any, parameters: SegmentationValidator):

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


class RegressionArray(Array):

    def create(self, source: str, parameters: RegressionValidator):

        array = np.array([float(source)])

        return array

    def preprocess(self, array: np.ndarray, preprocess: Any, parameters: RegressionValidator):

        orig_shape = array.shape
        array = array.reshape(-1, 1)
        array = preprocess.transform(array)
        array = array.reshape(orig_shape)

        return array


class DepthArray(Array):

    def create(self, source: list, parameters: DepthValidator):

        array = np.array(source)[:parameters.depth]
        array = np.expand_dims(array, 1)

        return array

    def preprocess(self, array: np.ndarray, preprocess: Any, parameters: DepthValidator):

        orig_shape = array.shape
        array = array.reshape(-1, 1)
        array = preprocess.transform(array)
        array = array.reshape(orig_shape)

        return array


class TrendArray(Array):

    def create(self, source: list, parameters: TrendValidator):

        source = np.array(source)
        first_val, second_val = source[0], source[1]
        if abs((second_val - first_val) / first_val) * 100 <= parameters.deviation:
            array = 0
        elif second_val > first_val:
            array = 1
        else:
            array = 2
        if parameters.one_hot_encoding:
            ohe_array = np.zeros((3,))
            ohe_array[array] = 1
            array = ohe_array.astype(np.uint8)

        return array

    def preprocess(self, array: np.ndarray, preprocess: Any, parameters: TrendValidator):

        return array
