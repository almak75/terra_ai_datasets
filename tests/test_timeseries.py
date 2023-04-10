import numpy as np

from terra_ai_datasets.create import TimeseriesDepth, TimeseriesTrend


def test_timeseries_depth():

    dataset = TimeseriesDepth(
        csv_path='./tests/dataset/shares/YNDX_1d.csv',
        train_size=0.7,
        inputs=["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"],
        outputs=["<LOW>", "<CLOSE>"],
        preprocessing="None",
        length=30,
        step=7,
        depth=10,
    )
    dataset.create(use_generator=True)

    for inp, out in dataset.dataset['train'].batch(4):
        assert inp["input_1"].shape == (4, 30, 4), 'Input shape does not match'
        assert out["output_1"].shape == (4, 10, 2), 'Output shape does not match'

        assert inp["input_1"].dtype == np.float64
        assert out["output_1"].dtype == np.float64

        break


def test_timeseries_trend():

    dataset = TimeseriesTrend(
        csv_path='shares/YNDX_1d.csv',
        train_size=0.7,
        inputs=["<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"],
        output="<LOW>",
        preprocessing="None",
        length=100,
        step=30,
        deviation=.5,
        one_hot_encoding=False
    )
    dataset.create()

    for inp, out in dataset.dataset['train'].batch(4):
        assert inp["input_1"].shape == (4, 100, 4), 'Input shape does not match'
        assert out["output_1"].shape == (4, ), 'Output shape does not match'

        assert inp["input_1"].dtype == np.float64
        assert out["output_1"].dtype == np.int32

        break
