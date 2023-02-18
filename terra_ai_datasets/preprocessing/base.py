import os

import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.preprocessing.text import Tokenizer

from terra_ai_datasets.preprocessing.terra_image_scaler import TerraImageScaler


class CreatePreprocessing(object):

    def __init__(self, dataset_path=None):

        self.dataset_path = dataset_path
        self.preprocessing = {}

    def load_preprocesses(self, put_data):

        for put in put_data.keys():
            self.preprocessing[put] = {}
            for col_name in put_data[put].keys():
                prep_path = os.path.join(self.dataset_path, 'preprocessing', str(put), f'{col_name}.gz')
                if os.path.isfile(prep_path):
                    preprocess_object = joblib.load(prep_path)
                    if repr(preprocess_object) in ['MinMaxScaler()', 'StandardScaler()']:
                        if 'clip' not in preprocess_object.__dict__.keys():
                            preprocess_object.clip = False
                    self.preprocessing[put].update([(col_name, preprocess_object)])
                else:
                    self.preprocessing[put].update([(col_name, None)])

    @staticmethod
    def create_min_max_scaler(options):

        scaler = MinMaxScaler(feature_range=(options['min_scaler'], options['max_scaler']))

        return scaler

    @staticmethod
    def create_standard_scaler(options):  # array=None,

        scaler = StandardScaler()

        return scaler

    @staticmethod
    def create_terra_image_scaler(options):

        scaler = TerraImageScaler(shape=(options['height'], options['width']),
                                  min_max=(options['min_scaler'], options['max_scaler']))

        return scaler

    @staticmethod
    def create_tokenizer(text_list: list, options):

        """

        Args:
            text_list: list
                Список слов для обучения токенайзера.
            options: Параметры токенайзера:
                       num_words: int
                           Количество слов для токенайзера.
                       filters: str
                           Символы, подлежащие удалению.
                       lower: bool
                           Перевод заглавных букв в строчные.
                       split: str
                           Символ разделения.
                       char_level: bool
                           Учёт каждого символа в качестве отдельного токена.
                       oov_token: str
                           В случае указания этот токен будет заменять все слова, не попавшие в
                           диапазон частотности слов 0 < num_words.

        Returns:
            Объект Токенайзер.

        """
        tokenizer = Tokenizer(**{'num_words': options['max_words_count'],
                                 'filters': options['filters'],
                                 'lower': options['lower'],
                                 'split': ' ',
                                 'char_level': options['char_level'],
                                 'oov_token': '<UNK>'
                                 }
                              )
        tokenizer.fit_on_texts(text_list)

        # if not options['put'] in self.preprocessing.keys():
        #     self.preprocessing[options['put']] = {}
        # self.preprocessing[options['put']].update([(options['cols_names'], tokenizer)])

        return tokenizer

    def inverse_data(self, options: dict):

        out_dict = {}
        for put_id, value in options.items():
            out_dict[put_id] = {}
            for col_name, array in value.items():
                if type(self.preprocessing[put_id][col_name]) == StandardScaler or \
                        type(self.preprocessing[put_id][col_name]) == MinMaxScaler:
                    out_dict[put_id].update({col_name: self.preprocessing[put_id][col_name].inverse_transform(array)})
                elif type(self.preprocessing[put_id][col_name]) == Tokenizer:
                    inv_tokenizer = {index: word for word, index in
                                     self.preprocessing[put_id][col_name].word_index.items()}
                    out_dict[put_id].update({col_name: ' '.join([inv_tokenizer[seq] for seq in array])})

        return out_dict
