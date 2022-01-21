import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

import dataset_info

tf.random.set_seed(1)

Z_SIZE = 512
BATCH_SIZE = 64

N_COLS = int(np.ceil(np.sqrt(BATCH_SIZE)))
N_ROWS = int(BATCH_SIZE / N_COLS)

INDICES = (-1, 1)
N_BATCHES = len(INDICES) * Z_SIZE / BATCH_SIZE

DATASET = dataset_info.celeba
model_version = 205

latent_vectors = []
latent_vector_label_batches = []
latent_vector_labels = []

for i in range(Z_SIZE):
    for j in INDICES:
        latent_vector = np.zeros(Z_SIZE)
        latent_vector[i] = j
        latent_vector = (latent_vector - np.mean(latent_vector)) / np.std(latent_vector)
        latent_vectors.append(latent_vector)
        if j > 0:
            j_sign = "+"
        else:
            j_sign = "-"

        latent_vector_labels.append(f"{i}{j_sign}")

        if len(latent_vector_labels) == BATCH_SIZE:
            latent_vector_label_batches.append(latent_vector_labels)
            latent_vector_labels = []

g_model = tf.keras.models.load_model(os.path.join(DATASET.name, "objects", f'g_model-{model_version:04d}.h5'))

ds_vectors = tf.data.Dataset.from_tensor_slices(latent_vectors).batch(BATCH_SIZE)

for batch_id, (latent_vector_batch, latent_label_batch) in enumerate(zip(ds_vectors, latent_vector_label_batches)):
    g_output = g_model(latent_vector_batch)
    images = (g_output.numpy() + 1) / 2

    fig = plt.figure(figsize=(N_COLS, 1.2 * N_ROWS), dpi=300, constrained_layout=True)

    for i, (image, label) in enumerate(zip(images, latent_label_batch)):
        ax = fig.add_subplot(N_ROWS, N_COLS, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image)
        ax.text(x=0.5, y=-0.1, s=label, size=10, horizontalalignment='center', verticalalignment='top',
                transform=ax.transAxes)

    filename = f'latent_vector_{model_version:03d}_batch_{batch_id:03d}.png'
    fig.savefig(os.path.join(DATASET.name, 'latent_vector', filename))
    plt.close(fig)
    print(f"Progress: {(batch_id + 1) / N_BATCHES:.0%}")
