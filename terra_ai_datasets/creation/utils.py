from enum import Enum
import logging
from typing import Union

import cv2
import librosa
import numpy as np
import pymorphy2
from pathlib import Path

from tensorflow.keras.preprocessing.text import text_to_word_sequence

from terra_ai_datasets.creation.validators.inputs import ImageProcessTypes, TextValidator, TextModeTypes, \
    ImageValidator, AudioValidator
from terra_ai_datasets.creation.validators.outputs import SegmentationValidator

logger_formatter = logging.Formatter(f"%(asctime)s | %(message)s", "%H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logger_formatter)

logger = logging.getLogger(__name__)
logger.addHandler(stream_handler)
logger.setLevel("INFO")


class DatasetFoldersData(Enum):
    arrays = "arrays"
    instructions = "instructions"
    preprocessing = "preprocessing"


class DatasetPutsData(Enum):
    inputs = "inputs"
    outputs = "outputs"


class DatasetInstructionFoldersData(Enum):
    dataframe = "dataframe"
    parameters = "parameters"


class DatasetFoldersPutsData:

    def __init__(self, native_path: Path):
        for folder in DatasetPutsData:
            native_path.joinpath(folder.value).mkdir(parents=True, exist_ok=True)
        self.native = native_path

    @property
    def inputs(self):
        return self.native.joinpath(DatasetPutsData.inputs.value)

    @property
    def outputs(self):
        return self.native.joinpath(DatasetPutsData.outputs.value)


class DatasetInstructionsFoldersData:

    def __init__(self, native_path: Path):
        for folder in DatasetInstructionFoldersData:
            native_path.joinpath(folder.value).mkdir(parents=True, exist_ok=True)
        self.native = native_path

    @property
    def dataframe(self):
        return self.native.joinpath(DatasetInstructionFoldersData.dataframe.value)

    @property
    def parameters(self):
        return self.native.joinpath(DatasetInstructionFoldersData.parameters.value)


class DatasetPathsData:

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.arrays = DatasetFoldersPutsData(root_path.joinpath(DatasetFoldersData.arrays.value))
        self.instructions = DatasetInstructionsFoldersData(root_path.joinpath(DatasetFoldersData.instructions.value))
        self.preprocessing = root_path.joinpath(DatasetFoldersData.preprocessing.value)
        self.preprocessing.mkdir(exist_ok=True)

    @property
    def config(self):
        return self.root_path.joinpath("config.json")


def apply_pymorphy(data_to_process: list):
    pymorphy = pymorphy2.MorphAnalyzer()

    data_to_return = []
    for text in data_to_process:
        words_list = text.split(' ')
        words_list = [pymorphy.parse(w)[0].normal_form for w in words_list]
        data_to_return.append(' '.join(words_list))

    return data_to_return


def extract_image_data(folder_path: Path, parameters: Union[ImageValidator, SegmentationValidator]):
    image_samples = []
    for image_path in folder_path.iterdir():
        image_samples.append(image_path)

    return image_samples


def extract_segmentation_data(folder_path: Path, parameters: SegmentationValidator):
    return extract_image_data(folder_path, parameters)


def extract_text_data(folder_path: Path, parameters: TextValidator):
    text_samples = []
    for text_path in folder_path.iterdir():
        with open(text_path, 'r', encoding="utf-8") as text_file:
            text = ' '.join(text_to_word_sequence(
                text_file.read().strip(), **{'lower': False, 'filters': parameters.filters, 'split': ' '})
            ).split()
        if parameters.mode == TextModeTypes.full:
            text_samples.append(' '.join(text[:parameters.max_words]))
        else:
            for i in range(0, len(text), parameters.step):
                text_sample = text[i: i + parameters.length]
                text_samples.append(' '.join(text_sample))
                if len(text_sample) < parameters.length:
                    break
    return text_samples


def extract_audio_data(folder_path: Path, parameters: AudioValidator):
    audio_samples = []
    for audio_path in folder_path.iterdir():
        if parameters.mode == TextModeTypes.full:
            audio_samples.append(';'.join([str(audio_path), f"0:{parameters.max_seconds}"]))
        else:
            duration = librosa.get_duration(filename=str(audio_path))
            start_idx, stop_idx = 0, parameters.length
            audio_samples.append(';'.join([str(audio_path), f"{start_idx}:{stop_idx}"]))
            while stop_idx < duration:
                start_idx += parameters.step
                stop_idx += parameters.step
                audio_samples.append(';'.join([str(audio_path), f"{start_idx}:{stop_idx}"]))
    return audio_samples


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
