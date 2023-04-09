import random

import matplotlib.pyplot as plt
import numpy as np

from terra_ai_datasets.creation.utils import create_put_array, preprocess_put_array


def visualize_image_classification(put_instructions, preprocessing):
    x_instructions = put_instructions[1]['1_Image']
    y_instructions = put_instructions[2]['2_Classification']

    classes_names = y_instructions.parameters.classes_names

    fig, ax = plt.subplots(1, len(classes_names), figsize=(3 * len(classes_names), 5))
    for i, cls_name in enumerate(classes_names):
        sample_idx = random.choice([idx for idx, cl_n in enumerate(y_instructions.data) if cl_n == cls_name])
        x_sample = x_instructions.data[sample_idx]
        y_sample = y_instructions.data[sample_idx]

        x_array = create_put_array(x_sample, x_instructions)
        y_array = create_put_array(y_sample, y_instructions)

        if preprocessing.get("1_Image"):
            x_array = preprocess_put_array(
                x_array, x_instructions, preprocessing["1_Image"]
            )

        ax[i].imshow(x_array)
        ax[i].set_title(f"{cls_name} - {str(y_array)}")

    plt.show()


def visualize_image_segmentation(put_instructions, preprocessing):
    sample_idx = random.randint(0, len(put_instructions[1]['1_Image'].data))

    x_sample = put_instructions[1]['1_Image'].data[sample_idx]
    y_sample = put_instructions[2]['2_Segmentation'].data[sample_idx]

    x_instructions = put_instructions[1]['1_Image']
    y_instructions = put_instructions[2]['2_Segmentation']

    x_array = create_put_array(x_sample, x_instructions)
    y_array = create_put_array(y_sample, y_instructions)

    if preprocessing.get("1_Image"):
        x_array = preprocess_put_array(
            x_array, x_instructions, preprocessing["1_Image"]
        )

    fig, ax = plt.subplots(1, 3, figsize=(5*3, 5))
    ax[0].imshow(x_array)
    ax[0].set_title(f"Вход")

    for i, (class_name, color) in enumerate(y_instructions.parameters.classes.items()):
        ax[i + 1].imshow(y_array[:, :, i])
        ax[i + 1].set_title(f"{class_name} - {str(color.as_rgb())}")

    plt.show()


def visualize_timeseries_depth(put_instructions, preprocessing):

    plt.figure(figsize=(15, 5))
    for col_name, col_data in put_instructions[1].items():
        x_instructions = put_instructions[1][col_name]
        x_array = create_put_array(col_data.data[:x_instructions.parameters.length], x_instructions)
        if preprocessing.get(col_name):
            x_array = preprocess_put_array(
                x_array, x_instructions, preprocessing[col_name]
            )
        plt.plot(x_array.flatten(), label=f"{col_name} (inp)")

    for col_name, col_data in put_instructions[2].items():
        y_instructions = put_instructions[2][col_name]
        y_array = create_put_array(
            col_data.data[y_instructions.parameters.length: + y_instructions.parameters.length + y_instructions.parameters.depth],
            y_instructions
        )
        if preprocessing.get(col_name):
            y_array = preprocess_put_array(
                y_array, y_instructions, preprocessing[col_name]
            )
        plt.plot([None for _ in range(col_data.parameters.length)] + y_array[:, 0].tolist(), label=f"{col_name} (out)")

    plt.grid(True, alpha=0.3)
    plt.legend()


def visualize_timeseries_trend(put_instructions, preprocessing):

    plt.figure(figsize=(15, 5))
    for col_name, col_data in put_instructions[1].items():
        x_instructions = put_instructions[1][col_name]
        x_array = create_put_array(col_data.data[:x_instructions.parameters.length], x_instructions)
        if preprocessing.get(col_name):
            x_array = preprocess_put_array(
                x_array, x_instructions, preprocessing[col_name]
            )
        plt.plot(x_array.flatten(), label=f"{col_name} (inp)")

    for col_name, col_data in put_instructions[2].items():
        y_instructions = put_instructions[2][col_name]
        y_array = create_put_array(
            col_data.data[y_instructions.parameters.length: + y_instructions.parameters.length + 2],
            y_instructions
        )
        if y_instructions.parameters.one_hot_encoding:
            y_array = np.argmax(y_array)
        if y_array == 1:
            color = "green"
        elif y_array == 2:
            color = "red"
        else:
            color = "cyan"
        start_idx, stop_idx = col_data.data[y_instructions.parameters.length: + y_instructions.parameters.length + 2]
        plt.plot([None for _ in range(y_instructions.parameters.length)] + [stop_idx], marker="o", markersize=2,
                 markeredgecolor=color, markerfacecolor=color)
        plt.arrow(y_instructions.parameters.length - 1, start_idx, 1, stop_idx - start_idx, width=0.2, color=color)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()


def visualize_text_classification(put_instructions, preprocessing):
    pass


def visualize_audio_classification(put_instructions, preprocessing):
    pass


def visualize_dataframe_classification(put_instructions, preprocessing):
    pass


def visualize_dataframe_regression(put_instructions, preprocessing):
    pass
