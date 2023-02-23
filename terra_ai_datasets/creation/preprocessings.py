import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TerraImageScaler:
    channels = ['red', 'green', 'blue']
    range = (0, 1)

    def __init__(self, height: int, width: int):

        self.shape: tuple = (height, width)
        self.trained_values: dict = {'red': {'min': np.full(self.shape, 255, dtype='uint8'),
                                             'max': np.zeros(self.shape, dtype='uint8')},
                                     'green': {'min': np.full(self.shape, 255, dtype='uint8'),
                                               'max': np.zeros(self.shape, dtype='uint8')},
                                     'blue': {'min': np.full(self.shape, 255, dtype='uint8'),
                                              'max': np.zeros(self.shape, dtype='uint8')}}

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

        return array.astype('uint8')


def create_min_max_scaler(**kwargs):

    scaler = MinMaxScaler()

    return scaler


def create_standard_scaler(**kwargs):

    scaler = StandardScaler()

    return scaler


def create_terra_image_scaler(**kwargs):

    scaler = TerraImageScaler(height=kwargs['height'], width=kwargs['width'])

    return scaler


# @staticmethod
# def create_tokenizer(text_list: list, options):
#     tokenizer = Tokenizer(**{'num_words': options['max_words_count'],
#                              'filters': options['filters'],
#                              'lower': options['lower'],
#                              'split': ' ',
#                              'char_level': options['char_level'],
#                              'oov_token': '<UNK>'
#                              }
#                           )
#     tokenizer.fit_on_texts(text_list)

#     return tokenizer


# def create_word2vec(text_list: list, options):
#
#     text_list = [elem.split(' ') for elem in text_list]
#     word2vec = Word2Vec(text_list, **{'size': options['size'],
#                                       'window': options['window'],
#                                       'min_count': options['min_count'],
#                                       'workers': options['workers'],
#                                       'iter': options['iter']
#                                       }
#                         )
#     return word2vec

# def create_tokenizer(text_list: list, options):
#     tokenizer = Tokenizer(**{'num_words': options['max_words_count'],
#                              'filters': options['filters'],
#                              'lower': options['lower'],
#                              'split': ' ',
#                              'char_level': options['char_level'],
#                              'oov_token': '<UNK>'
#                              }
#                           )
#     tokenizer.fit_on_texts(text_list)
#
#     # if not options['put'] in self.preprocessing.keys():
#     #     self.preprocessing[options['put']] = {}
#     # self.preprocessing[options['put']].update([(options['cols_names'], tokenizer)])
#
#     return tokenizer

# def create_word2vec(text_list: list, options):
#
#     text_list = [elem.split(' ') for elem in text_list]
#     word2vec = Word2Vec(text_list, **{'size': options['size'],
#                                       'window': options['window'],
#                                       'min_count': options['min_count'],
#                                       'workers': options['workers'],
#                                       'iter': options['iter']
#                                       }
#                         )
#
#     # if not options['put'] in self.preprocessing.keys():
#     #     self.preprocessing[options['put']] = {}
#     # self.preprocessing[options['put']].update([(options['cols_names'], word2vec)])
#
#     return word2vec

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
