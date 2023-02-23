from enum import Enum
import logging

import cv2
import numpy as np
from pathlib import Path

from terra_ai_datasets.creation.validators.inputs import ImageProcessTypes

logger_formatter = logging.Formatter(f"%(asctime)s | %(message)s", "%H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logger_formatter)

logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.setLevel("INFO")


class DatasetFoldersData(Enum):
    arrays = "arrays"
    instructions = "instructions"
    preprocessing = "preprocessing"


class DatasetPutsData(Enum):
    inputs = "inputs"
    outputs = "outputs"


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


class DatasetPathsData:

    def __init__(self, root_path: Path):
        for folder in DatasetFoldersData:
            # root_path.joinpath(folder.value).mkdir(parents=True, exist_ok=True)
            self.__dict__[folder.name] = DatasetFoldersPutsData(root_path.joinpath(folder.value))
        self.root_path = root_path
        # self.arrays = DatasetFoldersPutsData(self.root_path.joinpath(DatasetFoldersData.arrays.value))
        # self.instructions = DatasetFoldersPutsData(self.root_path.joinpath(DatasetFoldersData.instructions.value))
        # self.preprocessing = DatasetFoldersPutsData(self.root_path.joinpath(DatasetFoldersData.preprocessing.value))

    @property
    def config(self):
        return self.root_path.joinpath("config.json")


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
