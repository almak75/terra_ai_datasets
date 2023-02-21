import numpy as np
from PIL import Image

from terra_ai_datasets.creation.dataset import Array
from terra_ai_datasets.creation.utils import resize_frame
from terra_ai_datasets.creation.validators.inputs import ImageNetworkTypes, ImageValidator
from terra_ai_datasets.creation.validators.outputs import SegmentationValidator


class ImageArray(Array):

    def create(self, source: str, parameters: ImageValidator):

        image = Image.open(source)
        array = np.asarray(image)
        # frame_mode = options['image_mode'] if 'image_mode' in options.keys() else 'stretch'  # Временное решение
        # array = resize_frame(image_array=array,
        #                      target_shape=(parameters.height, parameters.width),
        #                      frame_mode=parameters.process)

        if parameters.network == ImageNetworkTypes.linear:
            array = array.reshape(np.prod(np.array(array.shape)))

        return array

    def preprocess(self, array: np.ndarray, **options):

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

    def preprocess(self, array: np.ndarray, **options):

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
