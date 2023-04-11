from typing import Any

import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from terra_ai_datasets.creation.validators.inputs import TextValidator, ImageValidator


class TerraImageScaler:
    channels = ['red', 'green', 'blue']
    range = (0, 1)

    def __init__(self, height: int, width: int):

        self.shape: tuple = (height, width)
        self.trained_values: dict = {ch: {'min': np.full(self.shape, 255, dtype='uint8'),
                                          'max': np.zeros(self.shape, dtype='uint8')} for ch in self.channels}

    def fit(self, img):
        for i, channel in enumerate(self.channels):
            min_mask = img[:, :, i] < self.trained_values[channel]['min']
            max_mask = img[:, :, i] > self.trained_values[channel]['max']
            self.trained_values[channel]['min'][min_mask] = img[:, :, i][min_mask]
            self.trained_values[channel]['max'][max_mask] = img[:, :, i][max_mask]

    def transform(self, img):

        transformed_img = []
        for ch in self.channels:
            x = img[:, :, self.channels.index(ch)]
            y1 = np.full(self.shape, self.range[0])
            y2 = np.full(self.shape, self.range[1])
            x1 = self.trained_values[ch]['min']
            x2 = self.trained_values[ch]['max']
            y = y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)
            transformed_img.append(y)

        array = np.moveaxis(np.array(transformed_img), 0, -1)

        array[array < self.range[0]] = self.range[0]
        array[array > self.range[1]] = self.range[1]

        if np.isnan(array).any():
            array = np.nan_to_num(array)

        return array

    def inverse_transform(self, img):

        transformed_img = []
        for ch in self.channels:
            x = img[:, :, self.channels.index(ch)]
            x1 = np.full(self.shape, self.range[0])
            x2 = np.full(self.shape, self.range[1])
            y1 = self.trained_values[ch]['min']
            y2 = self.trained_values[ch]['max']
            y = y1 + ((x - x1) / (x2 - x1)) * (y2 - y1)
            transformed_img.append(y)

        array = np.moveaxis(np.array(transformed_img), 0, -1)

        array[array < 0] = 0
        array[array > 255] = 255

        return array.astype(np.uint8)


def create_min_max_scaler(parameters: Any):

    scaler = MinMaxScaler()

    return scaler


def create_standard_scaler(parameters: Any):

    scaler = StandardScaler()

    return scaler


def create_terra_image_scaler(parameters: ImageValidator):

    scaler = TerraImageScaler(height=parameters.height, width=parameters.width)

    return scaler


def create_tokenizer(parameters: TextValidator):
    tokenizer = Tokenizer(**{'num_words': parameters.max_words_count,
                             'filters': parameters.filters,
                             'lower': True,
                             'split': ' ',
                             'char_level': False,
                             'oov_token': '<UNK>'}
                          )
    return tokenizer


def create_bag_of_words(parameters: TextValidator):
    return create_tokenizer(parameters)


def create_embedding(parameters: TextValidator):
    return create_tokenizer(parameters)


def create_word_to_vec(text_list: list, parameters: TextValidator):

    text_list = [elem.split(' ') for elem in text_list]
    word2vec = Word2Vec(text_list, vector_size=parameters.word2vec_size)
    return word2vec


# def inverse_data(self, options: dict):
#     out_dict = {}
#     for put_id, value in options.items():
#         out_dict[put_id] = {}
#         for col_name, array in value.items():
#             if type(self.preprocessing[put_id][col_name]) == StandardScaler or \
#                     type(self.preprocessing[put_id][col_name]) == MinMaxScaler:
#                 out_dict[put_id].update({col_name: self.preprocessing[put_id][col_name].inverse_transform(array)})
#
#             elif type(self.preprocessing[put_id][col_name]) == Tokenizer:
#                 inv_tokenizer = {index: word for word, index in
#                                  self.preprocessing[put_id][col_name].word_index.items()}
#                 out_dict[put_id].update({col_name: ' '.join([inv_tokenizer[seq] for seq in array])})
#
#             else:
#                 out_dict[put_id].update({col_name: ' '.join(
#                     [self.preprocessing[put_id][col_name].most_similar(
#                         positive=[seq], topn=1)[0][0] for seq in array])})
#     return out_dict
