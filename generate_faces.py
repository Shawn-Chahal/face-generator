import tensorflow as tf
import matplotlib.pyplot as plt
import os
import numpy as np

import dataset_info


def generate_composite(add_indices, random_seed):
    rng = np.random.default_rng(random_seed)
    g_model = tf.keras.models.load_model(os.path.join(DATASET.name, "objects", f'g_model-{MODEL_VERSION:04d}.h5'))
    fixed_z = [rng.normal(size=Z_SIZE) for i in range(BATCH_SIZE)]
    ds_z = tf.data.Dataset.from_tensor_slices(fixed_z).batch(BATCH_SIZE)

    for batch_z in ds_z:
        images = (g_model(batch_z) + 1) / 2

    n_rows = 8
    n_cols = 8

    fig = plt.figure(figsize=(n_cols, 1.2 * n_rows), dpi=300)
    for i in range(BATCH_SIZE):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 1.1, f'Face {i:02d}', size=10, horizontalalignment='center', verticalalignment='bottom',
                transform=ax.transAxes)
        ax.imshow(images[i])

    plt.tight_layout()
    plt.savefig(os.path.join(DATASET.name, 'testing', f'faces_{MODEL_VERSION:04d}.png'))
    plt.close(fig)

    n_rows = 1
    n_cols = len(add_indices) + 1

    fig = plt.figure(figsize=(n_cols, n_rows), dpi=300)

    for i in range(len(add_indices)):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 1.1, f'Face {add_indices[i]:02d}', size=10, horizontalalignment='center',
                verticalalignment='bottom',
                transform=ax.transAxes)
        if i < len(add_indices) - 1:
            operator = '+'
        else:
            operator = '='
        ax.text(1.1, 0.5, operator, size=16, horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes)
        ax.imshow(images[add_indices[i]])

    latent_sum = 0

    for i in add_indices:
        latent_sum = latent_sum + fixed_z[i]

    latent_mean = tf.math.reduce_mean(latent_sum)
    latent_std = tf.math.reduce_std(latent_sum)

    latent_sum = (latent_sum - latent_mean) / (latent_std + 0.000001)
    latent_sum = tf.reshape(latent_sum, (1, Z_SIZE))

    image_sum = (g_model(latent_sum) + 1) / 2

    ax = fig.add_subplot(n_rows, n_cols, n_rows * n_cols)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 1.1, f'Composite', size=10, horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)

    ax.imshow(image_sum[0])

    plt.tight_layout()
    plt.savefig(os.path.join(DATASET.name, 'testing', f'composite_{MODEL_VERSION:04d}.png'))
    plt.close(fig)


tf.random.set_seed(1)

MODEL_VERSION = 205
DATASET = dataset_info.celeba
Z_SIZE = 512
BATCH_SIZE = 64
GEN_DIM = 128
N_COLS = int(np.ceil(np.sqrt(BATCH_SIZE)))
N_ROWS = int(BATCH_SIZE / N_COLS)

generate_composite(add_indices=[7, 11, 44, 55, 56], random_seed=67)
