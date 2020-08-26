import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import os


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=channels)
    img = tf.image.convert_image_dtype(img, tf.float32)

    noise = 0.1

    img = tf.image.resize_with_crop_or_pad(img, input_dim, input_dim)
    img = tf.image.resize(img, size=(img_dim, img_dim))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=noise)
    img = tf.image.random_contrast(img, lower=(1 / (1 + noise)), upper=(1 + noise))

    if channels == 3:
        img = tf.image.random_saturation(img, lower=(1 / (1 + noise)), upper=(1 + noise))

    img = img * 2 - 1.0

    z_vector = tf.random.normal(shape=(z_size,))

    return z_vector, img


tf.random.set_seed(1)

input_dim = 178
channels = 3
hidden_dim = 4
kernel_size = 5
z_size = 128
n_filters = 512
img_dim = 64

lambda_gp = 10.0
beta_1 = 0.1
g_learning_rate = 0.00005
d_learning_rate = 0.0005

initial_epoch = 70
n_epochs = 5
batch_size = 64
status_frequency = 30
buffer_size = 4096

n_layers = int(np.around(np.log2(img_dim / hidden_dim)) + 0.0001)

list_ds = tf.data.Dataset.list_files(str(os.path.join('photos', 'img_align_celeba', '*.jpg')))
n_images = len(list(list_ds))
list_ds.shuffle(buffer_size=n_images, reshuffle_each_iteration=False)
ds = list_ds.map(process_path)
ds = ds.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

n_batches = int(n_images / batch_size)

if initial_epoch == 0:

    gen_model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=hidden_dim * hidden_dim * n_filters, input_shape=(z_size,), use_bias=False),
        tf.keras.layers.Reshape((hidden_dim, hidden_dim, n_filters)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()])

    for i in range(1, n_layers + 1):
        gen_model.add(
            tf.keras.layers.Conv2DTranspose(filters=int(n_filters / (2 ** i)), kernel_size=kernel_size, strides=2,
                                            padding='same', use_bias=False))
        gen_model.add(tf.keras.layers.BatchNormalization())
        gen_model.add(tf.keras.layers.LeakyReLU())

    gen_model.add(tf.keras.layers.Conv2DTranspose(filters=channels, kernel_size=kernel_size, padding='same',
                                                  use_bias=False, activation='tanh'))

    disc_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=int(n_filters / (2 ** n_layers)), kernel_size=kernel_size, padding='same',
                               input_shape=(img_dim, img_dim, channels)),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.LeakyReLU()])

    for i in range(n_layers - 1, -1, -1):
        disc_model.add(tf.keras.layers.Conv2D(filters=int(n_filters / (2 ** i)), kernel_size=kernel_size, strides=2,
                                              padding='same'))
        disc_model.add(tf.keras.layers.LayerNormalization())
        disc_model.add(tf.keras.layers.LeakyReLU())

    disc_model.add(tf.keras.layers.Flatten())
    disc_model.add(tf.keras.layers.Dense(1))

else:
    gen_model = tf.keras.models.load_model(os.path.join('objects', 'celeba', f'gen_model_{initial_epoch:03d}.h5'))
    disc_model = tf.keras.models.load_model(os.path.join('objects', 'celeba', f'disc_model_{initial_epoch:03d}.h5'))

with open(os.path.join('logs', 'celeba', f'gen_model_summary-{initial_epoch:03d}.txt'), 'w') as f_gen_model_summary:
    gen_model.summary(print_fn=(lambda x: f_gen_model_summary.write(f'{x}\n')))

gen_model.summary()

with open(os.path.join('logs', 'celeba', f'disc_model_summary-{initial_epoch:03d}.txt'), 'w') as f_disc_model_summary:
    disc_model.summary(print_fn=(lambda x: f_disc_model_summary.write(f'{x}\n')))

disc_model.summary()

g_optimizer = tf.keras.optimizers.Adam(learning_rate=g_learning_rate, beta_1=beta_1)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=d_learning_rate, beta_1=beta_1)

fixed_z = tf.random.normal(shape=(batch_size, z_size))

all_losses = []
epoch_samples = []
start_time = time.time()
status_time = time.time()

for epoch in range(1, n_epochs + 1):

    epoch_losses = []

    for i, (input_z, input_real) in enumerate(ds):

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            g_output = gen_model(input_z, training=True)

            d_critics_real = disc_model(input_real, training=True)
            d_critics_fake = disc_model(g_output, training=True)

            g_loss = -tf.math.reduce_mean(d_critics_fake)

            d_loss_real = -tf.math.reduce_mean(d_critics_real)
            d_loss_fake = tf.math.reduce_mean(d_critics_fake)

            with tf.GradientTape() as gp_tape:
                alpha = tf.random.uniform(shape=[d_critics_real.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)
                interpolated = (alpha * input_real + (1 - alpha) * g_output)
                gp_tape.watch(interpolated)
                d_critics_intp = disc_model(interpolated)

            grads_intp = gp_tape.gradient(d_critics_intp, [interpolated, ])[0]
            grads_intp_l2 = tf.sqrt(tf.reduce_sum(tf.square(grads_intp), axis=[1, 2, 3]))
            grad_penalty = tf.reduce_mean(tf.square(grads_intp_l2 - 1.0))

            d_loss_gp = lambda_gp * grad_penalty

            d_loss = d_loss_real + d_loss_fake + d_loss_gp

        d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
        d_optimizer.apply_gradients(grads_and_vars=zip(d_grads, disc_model.trainable_variables))

        g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)
        g_optimizer.apply_gradients(grads_and_vars=zip(g_grads, gen_model.trainable_variables))

        epoch_losses.append(
            (g_loss.numpy(), d_loss.numpy(), d_loss_real.numpy(), d_loss_fake.numpy(), d_loss_gp.numpy()))

        if time.time() > status_time + status_frequency:
            status_time = time.time()
            delta_t = time.time() - start_time
            hour = int(delta_t / 3600)
            minute = int((delta_t - 3600 * hour) / 60)
            second = int((delta_t - 3600 * hour - 60 * minute))

            print(f'Epoch: {epoch:2d}/{n_epochs} | Step: {i:5d}/{n_batches} | '
                  f'ET {hour:2d} h {minute:2d} min {second:2d} s | '
                  f'G loss: {g_loss.numpy():6.1f} | D loss: {d_loss.numpy():6.1f} '
                  f'(D-Real: {d_loss_real.numpy():6.1f}, D-Fake: {d_loss_fake.numpy():6.1f},'
                  f' D-GP: {d_loss_gp.numpy():6.1f})')

    all_losses.append(epoch_losses)
    mean_epoch_losses = np.mean(epoch_losses, axis=0)

    delta_t = time.time() - start_time
    hour = int(delta_t / 3600)
    minute = int((delta_t - 3600 * hour) / 60)
    second = int((delta_t - 3600 * hour - 60 * minute))

    print(f'Epoch: {epoch:2d}/{n_epochs} | Epoch Complete | '
          f'ET {hour:2d} h {minute:2d} min {second:2d} s | '
          f'G loss: {mean_epoch_losses[0]:6.1f} | D loss: {mean_epoch_losses[1]:6.1f} '
          f'(D-Real: {mean_epoch_losses[2]:6.1f}, D-Fake: {mean_epoch_losses[3]:6.1f},'
          f' D-GP: {mean_epoch_losses[4]:6.1f})')

    epoch_samples.append((gen_model(fixed_z, training=False).numpy() + 1) / 2.0)

    gen_model.save(os.path.join('objects', 'celeba', f'gen_model_{epoch + initial_epoch:03d}.h5'))
    disc_model.save(os.path.join('objects', 'celeba', f'disc_model_{epoch + initial_epoch:03d}.h5'))

n_rows = n_epochs + 0
n_cols = 8

fig = plt.figure(figsize=(6.5, 4), dpi=600)

ax = fig.add_subplot(1, 1, 1)
g_losses = [item[0] for item in itertools.chain(*all_losses)]
d_losses = [item[1] for item in itertools.chain(*all_losses)]
plt.plot(g_losses, label='Generator loss', alpha=0.75)
plt.plot(d_losses, label='Discriminator loss', alpha=0.75)
plt.legend()
ax.set_xlabel('Iteration')
ax.set_ylabel('Loss')

selected_epochs = np.arange(1, n_epochs + 1)

newpos = [(e * len(all_losses[-1])) for e in selected_epochs]
ax2 = ax.twiny()
ax2.set_xticks(newpos)
ax2.set_xticklabels(selected_epochs + initial_epoch)
ax2.xaxis.set_ticks_position('bottom')
ax2.xaxis.set_label_position('bottom')
ax2.spines['bottom'].set_position(('outward', 48))
ax2.set_xlabel('Epoch')
ax2.set_xlim(ax.get_xlim())
ax.tick_params(axis='both', which='major')
ax2.tick_params(axis='both', which='major')
plt.tight_layout()
plt.savefig(os.path.join('logs', 'celeba', f'learning_curve-{initial_epoch:03d}.png'))

fig = plt.figure(figsize=(n_cols, n_rows), dpi=600)
for i, e in enumerate(selected_epochs):
    for j in range(n_cols):
        ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.06, 0.5, f'Epoch {e + initial_epoch}', rotation=90, size=10, horizontalalignment='right',
                    verticalalignment='center', transform=ax.transAxes)

        if channels == 1:
            ax.imshow((epoch_samples[e - 1][j])[:, :, 0])
        else:
            ax.imshow((epoch_samples[e - 1][j])[:, :, :channels])

plt.tight_layout()
plt.savefig(os.path.join('logs', 'celeba', f'generated_faces-{initial_epoch:03d}.png'))
