import numpy as np

from terra_ai_datasets.create import ImageSegmentation


def test_segmentation_minmax_automatic():

    dataset = ImageSegmentation(source_path=['./airplane/Самолеты'], target_path=['./airplane/Сегменты'], train_size=.8,
                                width=64, height=80, preprocessing="MinMaxScaler", network="Convolutional",
                                process='Fit', rgb_range=50, num_classes=2)

    dataset.create()

    for inp, out in dataset.dataset['train'].batch(16):
        assert inp["input_1"].shape == (16, 80, 64, 3), 'Input shape does not match'
        assert out["output_1"].shape == (16, 80, 64, 2), 'Output shape does not match'

        assert inp["input_1"].dtype == np.float32
        assert out["output_1"].dtype == np.uint8

        break


def test_segmentation_no_scaler_from_file():
    dataset = ImageSegmentation(source_path=['./airplane/Самолеты'], target_path=['./airplane/Сегменты'], train_size=.8,
                                width=32, height=40, preprocessing="None", network="Convolutional",
                                process='Fit', rgb_range=50, classes_path='./airplane/labelmap.txt')

    dataset.create(use_generator=True)

    for inp, out in dataset.dataset['train'].batch(16):
        assert inp["input_1"].shape == (16, 40, 32, 3), 'Input shape does not match'
        assert out["output_1"].shape == (16, 40, 32, 2), 'Output shape does not match'

        assert inp["input_1"].dtype == np.uint8
        assert out["output_1"].dtype == np.uint8

        break
