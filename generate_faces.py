import tensorflow as tf
import matplotlib.pyplot as plt
import os

tf.random.set_seed(1)

z_size = 512
dim = 64
epoch = 440
batch_size = 64

n_rows = 8
n_cols = 8

fixed_z = tf.random.normal(shape=(batch_size, z_size))

gen_model = tf.keras.models.load_model(os.path.join('objects', f'gen_model_{dim:03d}-{epoch:03d}.h5'))

images = (gen_model(fixed_z) + 1) / 2

fig = plt.figure(figsize=(n_cols, 1.2 * n_rows), dpi=300)
for i in range(batch_size):
    ax = fig.add_subplot(n_rows, n_cols, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 1.1, f'Face {i:02d}', size=10, horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    ax.imshow(images[i])

plt.tight_layout()
plt.savefig(os.path.join('logs', 'testing', f'faces_{dim:03d}-{epoch:03d}.png'))

add_index = [54, 59, 62]

n_rows = 1
n_cols = len(add_index) + 1

fig = plt.figure(figsize=(n_cols, n_rows), dpi=300)

for i in range(len(add_index)):
    ax = fig.add_subplot(n_rows, n_cols, i + 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.5, 1.1, f'Face {add_index[i]:02d}', size=10, horizontalalignment='center', verticalalignment='bottom',
            transform=ax.transAxes)
    if i < len(add_index) - 1:
        operator = '+'
    else:
        operator = '='
    ax.text(1.1, 0.5, operator, size=16, horizontalalignment='left', verticalalignment='center',
            transform=ax.transAxes)
    ax.imshow(images[add_index[i]])

latent_sum = 0

for i in add_index:
    latent_sum = latent_sum + fixed_z[i]

latent_mean = tf.math.reduce_mean(latent_sum)
latent_std = tf.math.reduce_std(latent_sum)

latent_sum = (latent_sum - latent_mean) / (latent_std + 0.000001)
latent_sum = tf.reshape(latent_sum, (1, z_size))

image_sum = (gen_model(latent_sum) + 1) / 2

ax = fig.add_subplot(n_rows, n_cols, n_rows * n_cols)
ax.set_xticks([])
ax.set_yticks([])
ax.text(0.5, 1.1, f'Composite', size=10, horizontalalignment='center', verticalalignment='bottom',
        transform=ax.transAxes)

ax.imshow(image_sum[0])

plt.tight_layout()
plt.savefig(os.path.join('logs', 'testing', f'face-composite_{dim:03d}-{epoch:03d}.png'))
