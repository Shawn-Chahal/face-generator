import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

tf.random.set_seed(1)

z_size = 128
epoch = 65
max_sdev = 10
values = (-max_sdev, max_sdev)
n_images = z_size * 2
n_rows = 32
n_cols = int(n_images / n_rows)

gen_model = tf.keras.models.load_model(os.path.join('objects', f'gen_model_{epoch:03d}.h5'))

fig = plt.figure(figsize=(n_cols, 1.2 * n_rows), dpi=300)
for i in range(z_size):

    z_vector = np.zeros((2, z_size))
    z_vector[:, i] = values
    images = (gen_model(z_vector) + 1) / 2

    for j in range(2):
        row_index = (i % n_rows)
        column_index = 2 * int(i / n_rows) + j
        plot_index = row_index * n_cols + column_index + 1
        ax = fig.add_subplot(n_rows, n_cols, plot_index)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 1.1, f'x_{i} = {values[j]}', size=10, horizontalalignment='center', verticalalignment='bottom',
                transform=ax.transAxes)
        ax.imshow(images[j])

plt.tight_layout()
plt.savefig(os.path.join('logs', 'testing', f'latent_{epoch:03d}.png'))
