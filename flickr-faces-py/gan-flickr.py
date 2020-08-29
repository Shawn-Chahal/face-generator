import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
import os


def get_time(t):
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    t = t - 60 * minutes
    seconds = int(t)

    return hours, minutes, seconds


def load_network(dimensions, epoch):
    gen_model = tf.keras.models.load_model(os.path.join('objects', f'gen_model_{dimensions:03d}-{epoch:03d}.h5'))
    disc_model = tf.keras.models.load_model(os.path.join('objects', f'disc_model_{dimensions:03d}-{epoch:03d}.h5'))

    return gen_model, disc_model


def grow_network_from(dimensions, epoch):
    gen_model = tf.keras.models.load_model(os.path.join('objects', f'gen_model_{dimensions:03d}-{epoch:03d}.h5'))
    gen_inputs = gen_model.input
    gen_outputs = gen_model.layers[1](gen_inputs)
    for i in range(2, len(gen_model.layers) - 1):
        gen_outputs = gen_model.layers[i](gen_outputs)

    gen_outputs = tf.keras.layers.Conv2DTranspose(filters=FILTERS[2 * dimensions], kernel_size=KERNEL_SIZE,
                                                  padding='same', strides=2, use_bias=False,
                                                  name=f'conv2d_transpose_{2 * dimensions:03d}')(gen_outputs)

    gen_outputs = tf.keras.layers.BatchNormalization(name=f'batch_norm_{2 * dimensions:03d}')(gen_outputs)
    gen_outputs = tf.keras.layers.LeakyReLU(name=f'leaky_relu_{2 * dimensions:03d}')(gen_outputs)
    gen_outputs = tf.keras.layers.Conv2DTranspose(filters=CHANNELS, kernel_size=KERNEL_SIZE, padding='same',
                                                  use_bias=False, activation='tanh',
                                                  name=f'output_{2 * dimensions:03d}')(gen_outputs)

    new_gen_model = tf.keras.Model(inputs=gen_inputs, outputs=gen_outputs, name='gen_model')

    disc_model = tf.keras.models.load_model(os.path.join('objects', f'disc_model_{dimensions:03d}-{epoch:03d}.h5'))
    disc_inputs = tf.keras.Input(shape=((2 * dimensions), (2 * dimensions), CHANNELS), name='input')
    disc_outputs = tf.keras.layers.Conv2D(filters=FILTERS[2 * dimensions], kernel_size=KERNEL_SIZE, padding='same',
                                          name=f'conv2d_{2 * dimensions:03d}')(disc_inputs)

    disc_outputs = tf.keras.layers.LayerNormalization(name=f'layer_norm_{2 * dimensions:03d}')(disc_outputs)

    disc_outputs = tf.keras.layers.LeakyReLU(name=f'leaky_relu_{2 * dimensions:03d}')(disc_outputs)

    disc_outputs = tf.keras.layers.Conv2D(filters=FILTERS[dimensions], kernel_size=KERNEL_SIZE, padding='same',
                                          strides=2, name=f'conv2d_{dimensions:03d}')(disc_outputs)

    disc_outputs = disc_model.layers[2](disc_outputs)
    for i in range(3, len(disc_model.layers)):
        disc_outputs = disc_model.layers[i](disc_outputs)

    new_disc_model = tf.keras.Model(inputs=disc_inputs, outputs=disc_outputs, name='disc_model')

    return new_gen_model, new_disc_model


def create_base_network():
    output_dim = 4

    gen_inputs = tf.keras.Input(shape=(Z_SIZE,), name='input')
    gen_hidden = tf.keras.layers.Reshape((1, 1, Z_SIZE), name='reshape')(gen_inputs)
    gen_hidden = tf.keras.layers.Conv2DTranspose(filters=FILTERS[output_dim], kernel_size=output_dim, use_bias=False,
                                                 name=f'conv2d_transpose_{output_dim:03d}')(gen_hidden)

    gen_hidden = tf.keras.layers.BatchNormalization(name=f'batch_norm_{output_dim:03d}')(gen_hidden)
    gen_hidden = tf.keras.layers.LeakyReLU(name=f'leaky_relu_{output_dim:03d}')(gen_hidden)
    gen_outputs = tf.keras.layers.Conv2DTranspose(filters=CHANNELS, kernel_size=KERNEL_SIZE, padding='same',
                                                  use_bias=False, activation='tanh',
                                                  name=f'output_{output_dim:03d}')(gen_hidden)

    disc_inputs = tf.keras.Input(shape=(output_dim, output_dim, CHANNELS), name='input')
    disc_hidden = tf.keras.layers.Conv2D(filters=FILTERS[output_dim], kernel_size=KERNEL_SIZE, padding='same',
                                         name=f'conv2d_{output_dim:03d}')(disc_inputs)

    disc_hidden = tf.keras.layers.LayerNormalization(name=f'layer_norm_{output_dim:03d}')(disc_hidden)
    disc_hidden = tf.keras.layers.LeakyReLU(name=f'leaky_relu_{output_dim:03d}')(disc_hidden)
    disc_hidden = tf.keras.layers.Conv2D(filters=FILTERS[output_dim], kernel_size=output_dim,
                                         name=f'conv2d_{1:03d}')(disc_hidden)
    disc_hidden = tf.keras.layers.LayerNormalization(name=f'layer_norm_{1:03d}')(disc_hidden)
    disc_hidden = tf.keras.layers.LeakyReLU(name=f'leaky_relu_{1:03d}')(disc_hidden)
    disc_hidden = tf.keras.layers.Flatten(name='flatten')(disc_hidden)
    disc_outputs = tf.keras.layers.Dense(1, name='output')(disc_hidden)

    gen_model = tf.keras.Model(inputs=gen_inputs, outputs=gen_outputs, name="gen_model")
    disc_model = tf.keras.Model(inputs=disc_inputs, outputs=disc_outputs, name="disc_model")

    return gen_model, disc_model


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.resize(img, size=(gen_dim, gen_dim))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=NOISE)
    img = tf.image.random_contrast(img, lower=(1 / (1 + NOISE)), upper=(1 + NOISE))
    img = tf.image.random_saturation(img, lower=(1 / (1 + NOISE)), upper=(1 + NOISE))

    img = img * 2 - 1.0

    z_vector = tf.random.normal(shape=(Z_SIZE,))

    return z_vector, img


tf.random.set_seed(1)

NOISE = 0.1
CHANNELS = 3
KERNEL_SIZE = 5
Z_SIZE = 512
FILTERS = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}

LAMBDA_GP = 10.0
BETA_1 = 0.0
G_LR = 0.00005  # Generator learning rate
D_LR = 0.0005  # Discriminator learning rate

BATCH_SIZE = 32
BUFFER_SIZE = 4096
STATUS_FREQUENCY = 30  # seconds

grow_network = False
gen_dim = 32
initial_epoch = 145
n_epochs = 35

fixed_z = tf.random.normal(shape=(BATCH_SIZE, Z_SIZE))

list_ds = tf.data.Dataset.list_files(str(os.path.join('..', 'photos', 'thumbnails128x128', '*', '*.png')))
n_images = len(list(list_ds))
list_ds.shuffle(buffer_size=n_images, reshuffle_each_iteration=False)
ds = list_ds.map(process_path)
ds = ds.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

n_batches = int(n_images / BATCH_SIZE)

if initial_epoch == 0:
    gen_model, disc_model = create_base_network()

elif grow_network:
    gen_model, disc_model = grow_network_from(gen_dim // 2, initial_epoch)

else:
    gen_model, disc_model = load_network(gen_dim, initial_epoch)

with open(os.path.join('logs', 'training', f'gen_model_summary_{gen_dim:03d}-{initial_epoch:03d}.txt'),
          'w') as f_gen_model_summary:
    gen_model.summary(print_fn=(lambda x: f_gen_model_summary.write(f'{x}\n')))

gen_model.summary()

with open(os.path.join('logs', 'training', f'disc_model_summary_{gen_dim:03d}-{initial_epoch:03d}.txt'),
          'w') as f_disc_model_summary:
    disc_model.summary(print_fn=(lambda x: f_disc_model_summary.write(f'{x}\n')))

disc_model.summary()

g_optimizer = tf.keras.optimizers.Adam(learning_rate=G_LR, beta_1=BETA_1)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LR, beta_1=BETA_1)

all_losses = []
epoch_samples = []
start_time = time.time()
status_time = time.time()

for epoch in range(1, n_epochs + 1):

    epoch_losses = []
    last_step = -1
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

            d_loss_gp = LAMBDA_GP * grad_penalty

            d_loss = d_loss_real + d_loss_fake + d_loss_gp

        d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
        d_optimizer.apply_gradients(grads_and_vars=zip(d_grads, disc_model.trainable_variables))

        g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)
        g_optimizer.apply_gradients(grads_and_vars=zip(g_grads, gen_model.trainable_variables))

        epoch_losses.append(
            (g_loss.numpy(), d_loss.numpy(), d_loss_real.numpy(), d_loss_fake.numpy(), d_loss_gp.numpy()))

        if time.time() - status_time > STATUS_FREQUENCY:
            seconds_per_step = (time.time() - status_time) / (i - last_step)
            steps_remaining = (n_batches - 1) - i
            epochs_remaining = n_epochs - epoch
            seconds_remaining = seconds_per_step * (steps_remaining + n_batches * epochs_remaining)
            status_time = time.time()
            elapsed_time = get_time(time.time() - start_time)
            time_remaining = get_time(seconds_remaining)

            last_step = i
            print(f'Epoch: {epoch:2d}/{n_epochs} | Step: {i:5d}/{n_batches} | '
                  f'Time Remaining: {time_remaining[0]:2d} h {time_remaining[1]:2d} min {time_remaining[2]:2d} s | '
                  f'Elapsed Time {elapsed_time[0]:2d} h {elapsed_time[1]:2d} min {elapsed_time[2]:2d} s | '
                  f'G loss: {g_loss.numpy():6.1f} | D loss: {d_loss.numpy():6.1f} '
                  f'(D-Real: {d_loss_real.numpy():6.1f}, D-Fake: {d_loss_fake.numpy():6.1f},'
                  f' D-GP: {d_loss_gp.numpy():6.1f})')

    all_losses.append(epoch_losses)
    mean_epoch_losses = np.mean(epoch_losses, axis=0)

    elapsed_time = get_time(time.time() - start_time)

    print(f'Epoch: {epoch:2d}/{n_epochs} | Epoch Complete | '
          f'Elapsed Time {elapsed_time[0]:2d} h {elapsed_time[1]:2d} min {elapsed_time[2]:2d} s | '
          f'G loss: {mean_epoch_losses[0]:6.1f} | D loss: {mean_epoch_losses[1]:6.1f} '
          f'(D-Real: {mean_epoch_losses[2]:6.1f}, D-Fake: {mean_epoch_losses[3]:6.1f},'
          f' D-GP: {mean_epoch_losses[4]:6.1f})')

    epoch_samples.append((gen_model(fixed_z, training=False).numpy() + 1) / 2.0)

    gen_model.save(os.path.join('objects', f'gen_model_{gen_dim:03d}-{epoch + initial_epoch:03d}.h5'))
    disc_model.save(os.path.join('objects', f'disc_model_{gen_dim:03d}-{epoch + initial_epoch:03d}.h5'))

n_rows = n_epochs + 0
n_cols = 8

fig = plt.figure(figsize=(6.5, 4), dpi=300)

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
plt.savefig(os.path.join('logs', 'training', f'learning_curve_{gen_dim:03d}-{initial_epoch:03d}.png'))

fig = plt.figure(figsize=(n_cols, n_rows), dpi=300)
for i, e in enumerate(selected_epochs):
    for j in range(n_cols):
        ax = fig.add_subplot(n_rows, n_cols, i * n_cols + j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(-0.06, 0.5, f'Epoch {e + initial_epoch}', rotation=90, size=10, horizontalalignment='right',
                    verticalalignment='center', transform=ax.transAxes)

        if CHANNELS == 1:
            ax.imshow((epoch_samples[e - 1][j])[:, :, 0])
        else:
            ax.imshow((epoch_samples[e - 1][j])[:, :, :CHANNELS])

plt.tight_layout()
plt.savefig(os.path.join('logs', 'training', f'generated_faces_{gen_dim:03d}-{initial_epoch:03d}.png'))
