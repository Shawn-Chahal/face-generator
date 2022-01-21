import os

import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import dataset_info


def generate_z_vector():
    def mask_dimension(dimension, value):
        mask_d = df.loc[:, "Dimension"] == dimension

        if value > 0:
            mask_s = df.loc[:, "Sign"] == "+"
        else:
            mask_s = df.loc[:, "Sign"] == "-"

        mask_value = df.loc[mask_d * mask_s, "Mask"].tolist()[0]

        return mask_value == 1

    z_vector = rng.normal(size=Z_SIZE)

    for d, v in enumerate(z_vector):
        if mask_dimension(d, v):
            z_vector[d] = 0

    z_vector = (z_vector - np.mean(z_vector)) / np.std(z_vector)

    return z_vector


def generate_batch_images(ds_z, filename_suffix):
    for batch_z in ds_z:
        g_output = g_model(batch_z)

        images = (g_output.numpy() + 1) / 2

        fig = plt.figure(figsize=(N_COLS, 1.2 * N_ROWS), dpi=300, constrained_layout=True)

        for i, image in enumerate(images):
            ax = fig.add_subplot(N_ROWS, N_COLS, i + 1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(image)

        filename = f'latent_vector_adjustment_{model_version:03d}_{filename_suffix}.png'
        fig.savefig(os.path.join(DATASET.name, 'manual_info', filename))
        plt.close(fig)


pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

tf.random.set_seed(1)
rng = np.random.default_rng(1)

DATASET = dataset_info.celeba

df = pd.read_csv(os.path.join(DATASET.name, "manual_info", "latent_vector_info.csv"))

Z_SIZE = 512
BATCH_SIZE = 64
N_COLS = int(np.ceil(np.sqrt(BATCH_SIZE)))
N_ROWS = int(BATCH_SIZE / N_COLS)

model_version = 205
g_model = tf.keras.models.load_model(os.path.join(DATASET.name, "objects", f'g_model-{model_version:04d}.h5'))

batch_z = [generate_z_vector(["X"]) for i in range(BATCH_SIZE)]

ds_z = tf.data.Dataset.from_tensor_slices(batch_z).batch(BATCH_SIZE)

generate_batch_images(ds_z, "adjusted")
