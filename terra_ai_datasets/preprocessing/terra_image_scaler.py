import numpy as np


class TerraImageScaler:

    def __init__(self, shape=(176, 240), min_max=(0, 1)):

        self.shape: tuple = shape
        self.trained_values: dict = {'red': {'min': np.full(shape, 255, dtype='uint8'),
                                             'max': np.zeros(shape, dtype='uint8')},
                                     'green': {'min': np.full(shape, 255, dtype='uint8'),
                                               'max': np.zeros(shape, dtype='uint8')},
                                     'blue': {'min': np.full(shape, 255, dtype='uint8'),
                                              'max': np.zeros(shape, dtype='uint8')}}
        self.range = min_max

        pass

    def fit(self, img):
        for i, channel in enumerate(['red', 'green', 'blue']):
            min_mask = img[:, :, i] < self.trained_values[channel]['min']
            max_mask = img[:, :, i] > self.trained_values[channel]['max']
            self.trained_values[channel]['min'][min_mask] = img[:, :, i][min_mask]
            self.trained_values[channel]['max'][max_mask] = img[:, :, i][max_mask]

    def transform(self, img):

        transformed_img = self.base_transform(img)
        array = np.moveaxis(np.array(transformed_img), 0, -1)

        array[array < self.range[0]] = self.range[0]
        array[array > self.range[1]] = self.range[1]

        return array

    def inverse_transform(self, img):

        transformed_img = self.base_transform(img)
        array = np.moveaxis(np.array(transformed_img), 0, -1)

        array[array < 0] = 0
        array[array > 255] = 255

        return array.astype('uint8')

    def base_transform(self, image):

        channels = ['red', 'green', 'blue']
        transformed_img = []
        for ch in channels:
            x = image[:, :, channels.index(ch)]
            x1 = np.full(self.shape, self.range[0])
            x2 = np.full(self.shape, self.range[1])
            y1 = self.trained_values[ch]['min']
            y2 = self.trained_values[ch]['max']
            y = y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)
            transformed_img.append(y)

        return transformed_img
