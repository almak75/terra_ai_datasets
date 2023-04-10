import numpy as np

from terra_ai_datasets.create import ImageClassification, TextClassification, AudioClassification, \
    DataframeClassification


def test_image_classification_minmax():

    dataset = ImageClassification(source_path=['./tests/dataset/cars/Мерседес', './tests/dataset/cars/Феррари'],
                                  width=80, height=64, preprocessing='MinMaxScaler', network="Linear", process='Fit',
                                  train_size=0.7, one_hot_encoding=True)

    dataset.create(use_generator=True)

    for inp, out in dataset.dataset['train'].batch(4):
        assert inp["input_1"].shape == (4, 64*80*3), 'Input shape does not match'
        assert out["output_1"].shape == (4, 2), 'Output shape does not match'

        assert inp["input_1"].dtype == np.float32, 'Input outputs wrong datatype'
        assert out["output_1"].dtype == np.uint8, 'Output outputs wrong datatype'

        break


def test_image_classification_terra_image():
    dataset = ImageClassification(source_path=['./tests/dataset/cars'], width=160, height=128,
                                  preprocessing='TerraImageScaler', network="Convolutional", process='Stretch',
                                  train_size=0.7, one_hot_encoding=False)

    dataset.create()

    for inp, out in dataset.dataset['train'].batch(6):
        assert inp["input_1"].shape == (6, 128, 160, 3), 'Input shape does not match'
        assert out["output_1"].shape == (6, ), 'Output shape does not match'

        assert inp["input_1"].dtype == np.float32
        assert out["output_1"].dtype == np.uint8

        break


def test_image_classification_no_scaler():
    dataset = ImageClassification(source_path=['./tests/dataset/cars/Мерседес', './tests/dataset/cars/Рено',
                                               './tests/dataset/cars/Феррари'], width=80, height=64,
                                  preprocessing='None', network="Convolutional", process='Cut',
                                  train_size=0.7, one_hot_encoding=True)

    dataset.create()

    for inp, out in dataset.dataset['train'].batch(6):
        assert inp["input_1"].shape == (6, 64, 80, 3), 'Input shape does not match'
        assert out["output_1"].shape == (6, 3), 'Output shape does not match'

        assert inp["input_1"].dtype == np.uint8
        assert out["output_1"].dtype == np.uint8

        break


def test_text_classification_emb():

    dataset = TextClassification(
        source_path=['./tests/dataset/symptoms/Аппендицит', './tests/dataset/symptoms/Гастрит',
                     './tests/dataset/symptoms/Гепатит'],
        train_size=0.7,
        max_words_count=20000,
        mode="Length and step",
        preprocessing="Embedding",
        length=100,
        step=30,
        one_hot_encoding=True,
        pymorphy=True
    )
    dataset.create()

    for inp, out in dataset.dataset['train'].batch(2):
        assert inp["input_1"].shape == (2, 100), 'Input shape does not match'
        assert out["output_1"].shape == (2, 3), 'Output shape does not match'

        assert inp["input_1"].dtype == np.int32
        assert out["output_1"].dtype == np.uint8

        break


def test_text_classification_bow():

    dataset = TextClassification(
        source_path=['./symptoms'],
        train_size=0.7,
        max_words_count=10000,
        mode="Full",
        preprocessing="Bag of words",
        max_words=2000,
        one_hot_encoding=True,
        pymorphy=False
    )

    dataset.create(use_generator=True)

    for inp, out in dataset.dataset['train'].batch(3):
        assert inp["input_1"].shape == (3, 10000), 'Input shape does not match'
        assert out["output_1"].shape == (3, 10), 'Output shape does not match'

        assert inp["input_1"].dtype == np.int8
        assert out["output_1"].dtype == np.uint8

        break


def test_text_classification_w2v():

    dataset = TextClassification(
        source_path=['./symptoms'],
        train_size=0.7,
        max_words_count=10000,
        word2vec_size=50,
        mode="Length and step",
        preprocessing="Word2Vec",
        length=100,
        step=30,
        one_hot_encoding=True,
        pymorphy=False
    )

    dataset.create(use_generator=True)

    for inp, out in dataset.dataset['train'].batch(16):
        assert inp["input_1"].shape == (16, 100, 50), 'Input shape does not match'
        assert out["output_1"].shape == (16, 10), 'Output shape does not match'

        assert inp["input_1"].dtype == np.float64
        assert out["output_1"].dtype == np.uint8

        break


def test_audio_classification():

    dataset = AudioClassification(
        source_path=['./smarthome/1_Кондиционер', './smarthome/2_Свет'],
        train_size=0.7,
        sample_rate=22050,
        mode="Length and step",
        parameter=["Audio signal", "MFCC", "RMS"],
        fill_mode="Last millisecond",
        resample="Kaiser fast",
        preprocessing="None",
        max_seconds=2,
        length=0.5,
        step=0.4,
        one_hot_encoding=True
        )

    dataset.create(use_generator=True)

    for inp, out in dataset.dataset['train'].batch(16):
        assert inp["input_1"].shape == (16, 11025), 'Input shape does not match'
        assert inp["input_2"].shape == (16, 22, 20), 'Input shape does not match'
        assert inp["input_3"].shape == (16, 22), 'Input shape does not match'
        assert out["output_1"].shape == (16, 2), 'Output shape does not match'

        assert inp["input_1"].dtype == np.float32
        assert inp["input_2"].dtype == np.float32
        assert inp["input_3"].dtype == np.float32
        assert out["output_1"].dtype == np.uint8

        break


def test_dataframe_classification():
    dataset = DataframeClassification(
        csv_path='./tests/dataset/flats/flats_cut.csv',
        train_size=0.5,
        inputs=[
            {"columns": ["Станция метро", "Пешком или на транспорте"], "type": "Categorical", "parameters": {"one_hot_encoding": True}},
            {"columns": ["Этаж"], "type": "Raw", "parameters": {}},
            {"columns": ["Примечание"], "type": "Text", "parameters": {"max_words_count": 20, "mode": "Full", "preprocessing": "Embedding", "max_words": 10, "pymorphy": False}},
        ],
        output="Тип дома",
        one_hot_encoding=False
    )
    dataset.create()

    for inp, out in dataset.dataset['train'].batch(4):
        assert inp["input_1"].shape == (4, 11), 'Input shape does not match'
        assert inp["input_2"].shape == (4, ), 'Input shape does not match'
        assert inp["input_3"].shape == (4, 10), 'Input shape does not match'
        assert out["output_1"].shape == (4, ), 'Output shape does not match'

        assert inp["input_1"].dtype == np.uint8
        assert inp["input_2"].dtype == np.int32
        assert inp["input_3"].dtype == np.int32
        assert out["output_1"].dtype == np.uint8

        break
