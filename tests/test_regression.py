import numpy as np

from terra_ai_datasets.create import DataframeRegression


def test_regression():

    dataset = DataframeRegression(
        csv_path='./tests/dataset/flats/flats_cut.csv',
        train_size=0.7,
        inputs=[
            {"columns": ["Станция метро", "Пешком или на транспорте"], "type": "Categorical", "parameters": {"one_hot_encoding": True}},
            {"columns": ["Этаж"], "type": "Raw", "parameters": {}},
            {"columns": ["Примечание"], "type": "Text", "parameters": {"max_words_count": 20000, "mode": "Full", "preprocessing": "Embedding", "max_words": 100, "pymorphy": False}},
        ],
        output="Цена, тыс.руб.",
        preprocessing="None"
    )
    dataset.create(use_generator=True)

    for inp, out in dataset.dataset['train'].batch(4):
        assert inp[1].shape == (4, 11), 'Input shape does not match'
        assert inp[2].shape == (4, ), 'Input shape does not match'
        assert inp[3].shape == (4, 100), 'Input shape does not match'
        assert out[4].shape == (4, ), 'Output shape does not match'

        assert inp[1].dtype == np.uint8
        assert inp[2].dtype == np.int32
        assert inp[3].dtype == np.int32
        assert out[4].dtype == np.float64

        break
