import os
import time
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import dataset_info


def process_image_path(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img, channels=CHANNELS, dtype=tf.float32)

    if DATASET.name is dataset_info.celeba.name:
        img = tf.image.resize_with_crop_or_pad(img, 178, 178)

    img = tf.image.resize(img, size=(GEN_DIM, GEN_DIM))
    img = tf.image.random_flip_left_right(img)
    img = tf.math.multiply(img, 2)
    img = tf.math.subtract(img, 1)

    z_vector = tf.random.normal(shape=(Z_SIZE,), seed=1)

    return z_vector, img


def create_network():
    def summarize_models():
        with open(os.path.join(DATASET.name, S_LOGS, f"gen_model_summary.txt"), "w") as f_gen_model_summary:
            nn_gen_model.summary(print_fn=(lambda x: f_gen_model_summary.write(f"{x}\n")))

        nn_gen_model.summary()

        with open(os.path.join(DATASET.name, S_LOGS, f"disc_model_summary.txt"), "w") as f_disc_model_summary:
            nn_disc_model.summary(print_fn=(lambda x: f_disc_model_summary.write(f"{x}\n")))

        nn_disc_model.summary()

    def generator_basic_block(block_input, filters, kernel_size=KERNEL_SIZE, padding="same", strides=1):
        block_output = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                                                       padding=padding, use_bias=False,
                                                       strides=strides)(block_input)
        block_output = tf.keras.layers.BatchNormalization()(block_output)
        block_output = tf.keras.layers.LeakyReLU()(block_output)
        return block_output

    def generator_to_rgb_block(block_input):
        block_output = tf.keras.layers.Conv2DTranspose(filters=CHANNELS, kernel_size=KERNEL_SIZE_RGB, padding="same",
                                                       use_bias=False)(block_input)
        return block_output

    def discriminator_basic_block(block_input, filters, kernel_size=KERNEL_SIZE, padding="same", strides=1):
        block_output = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                              padding=padding, strides=strides)(block_input)
        block_output = tf.keras.layers.LayerNormalization()(block_output)
        block_output = tf.keras.layers.LeakyReLU()(block_output)
        return block_output

    def discriminator_from_rgb_block(block_input):
        block_output = discriminator_basic_block(block_input, filters=FILTERS[output_dim], kernel_size=KERNEL_SIZE_RGB)

        return block_output

    min_dim = 4
    output_dim = min_dim + 0

    """GENERATOR"""

    gen_inputs = tf.keras.Input(shape=(Z_SIZE,))
    gen_hidden = tf.keras.layers.Reshape((1, 1, Z_SIZE))(gen_inputs)
    gen_hidden = generator_basic_block(gen_hidden, filters=FILTERS[output_dim], kernel_size=output_dim, padding="valid")

    gen_outputs = generator_to_rgb_block(gen_hidden)

    while output_dim < GEN_DIM:
        output_dim *= 2
        gen_hidden = generator_basic_block(gen_hidden, filters=FILTERS[output_dim], strides=2)
        gen_rgb_skip = generator_to_rgb_block(gen_hidden)
        gen_outputs = tf.keras.layers.UpSampling2D(interpolation="bilinear")(gen_outputs)
        gen_outputs = tf.keras.layers.Add()([gen_outputs, gen_rgb_skip])

    gen_outputs = tf.keras.layers.Activation("tanh")(gen_outputs)

    nn_gen_model = tf.keras.Model(inputs=gen_inputs, outputs=gen_outputs)

    """DISCRIMINATOR"""

    disc_input = tf.keras.Input(shape=(output_dim, output_dim, CHANNELS))
    disc_input_skip = disc_input
    disc_outputs = discriminator_from_rgb_block(disc_input)

    while output_dim > min_dim:
        output_dim = output_dim // 2

        disc_outputs = tf.keras.layers.Conv2D(filters=FILTERS[output_dim], kernel_size=KERNEL_SIZE, padding="same",
                                              strides=2)(disc_outputs)

        disc_input_skip = tf.keras.layers.AveragePooling2D(padding='same')(disc_input_skip)
        disc_from_rgb = tf.keras.layers.Conv2D(filters=FILTERS[output_dim], kernel_size=KERNEL_SIZE_RGB, padding="same",
                                               strides=1)(disc_input_skip)

        disc_outputs = tf.keras.layers.Add()([disc_outputs, disc_from_rgb])
        disc_outputs = tf.keras.layers.LayerNormalization()(disc_outputs)
        disc_outputs = tf.keras.layers.LeakyReLU()(disc_outputs)

    disc_outputs = discriminator_basic_block(disc_outputs, filters=FILTERS[output_dim], kernel_size=output_dim)
    disc_outputs = tf.keras.layers.Flatten()(disc_outputs)
    disc_outputs = tf.keras.layers.Dense(units=1)(disc_outputs)

    nn_disc_model = tf.keras.Model(inputs=disc_input, outputs=disc_outputs)

    """FINISH"""

    summarize_models()

    return nn_gen_model, nn_disc_model


def load_network():
    nn_gen_model = tf.keras.models.load_model(get_model_path(S_G))
    nn_disc_model = tf.keras.models.load_model(get_model_path(S_D))

    return nn_gen_model, nn_disc_model


def get_model_path(model_type):
    return os.path.join(DATASET.name, S_OBJECTS, f"{model_type}_model-{model_version:04d}.h5")


def get_print_time(t):
    days = int(t / 86400)
    t = t - 86400 * days
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    t = t - 60 * minutes
    seconds = int(t)

    return ReadableTime(days, hours, minutes, seconds)


def plot_learning_curve():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4), dpi=600, constrained_layout=True)
    ax.plot(dict_loss["Images trained"], dict_loss["Loss (G)"], label="Generator")
    ax.plot(dict_loss["Images trained"], dict_loss["Loss (D)"], label="Discriminator")
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_xlabel("Images trained")
    ax.set_ylabel("Loss")

    fig.savefig(os.path.join(DATASET.name, S_LOGS, "learning_curve.png"))
    plt.close(fig)


def plot_dashboard():
    fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(20, 4), dpi=600, constrained_layout=True)
    axs = axs.ravel()

    labels = ["Loss (G)", "Loss (D)", "Loss (D-Real)", "Loss (D-Fake)", "Loss (D-GP)"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for ax, label, color in zip(axs, labels, colors):
        ax.plot(dict_loss["Images trained"], dict_loss[label], color=color)
        ax.set_xlim(left=0)
        ax.set_xlabel("Images trained")
        ax.set_ylabel(label)

    fig.savefig(os.path.join(DATASET.name, S_LOGS, "learning_curve_dashboard.png"))
    plt.close(fig)


def plot_generator_images():
    gen_images = g_model(FIXED_Z)
    image_batch = (gen_images.numpy() + 1) / 2

    n_cols = int(np.ceil(np.sqrt(BATCH_SIZE)))
    n_rows = int(BATCH_SIZE / n_cols)

    fig = plt.figure(figsize=(n_cols, n_rows), dpi=300, constrained_layout=True)
    for i in range(n_cols * n_rows):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image_batch[i])

    fig.savefig(os.path.join(DATASET.name, S_GENERATED_FACES_LOCAL, f"generated_faces_{model_version:04d}.png"))
    fig.savefig(os.path.join(DATASET.name, S_LOGS, f"generated_faces_latest.png"))

    if model_version % LOG_FREQUENCY_GIT == 0:
        fig.savefig(os.path.join(DATASET.name, S_GENERATED_FACES, f"generated_faces_{model_version:04d}.png"))

    plt.close(fig)


tf.random.set_seed(1)

ReadableTime = namedtuple('ReadableTime', ['days', 'hours', 'minutes', 'seconds'])

S_LOGS = "logs"
S_OBJECTS = "objects"
S_GENERATED_FACES_LOCAL = "generated_faces_local"
S_GENERATED_FACES = "generated_faces"
S_G = "g"
S_D = "d"

LOG_FREQUENCY = 12 * 60  # seconds
LOG_FREQUENCY_GIT = 20  # versions

N_TRAINING_IMAGES = 10 ** 7
BUFFER_SIZE = 4096
BATCH_SIZE = 16
GEN_DIM = 128
CHANNELS = 3
KERNEL_SIZE = 5
KERNEL_SIZE_RGB = 1
FILTERS = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 32}
Z_SIZE = 512
LAMBDA_GP = 10
BETA_1 = 0
LEARNING_RATE = 0.0001

FIXED_Z = tf.random.normal(shape=(BATCH_SIZE, Z_SIZE))

""" TRAINING """

DATASET = dataset_info.celeba
model_version = 209

list_ds = tf.data.Dataset.list_files(DATASET.glob)
list_ds = list_ds.shuffle(buffer_size=len(list(list_ds)), reshuffle_each_iteration=False)
ds = list_ds.map(process_image_path)
ds = ds.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()

g_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)
d_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, beta_1=BETA_1)

if model_version == 0:
    g_model, d_model = create_network()

    dict_loss = {"Model version": [], "Images trained": [], "Time [s]": [], "Loss (G)": [], "Loss (D)": [],
                 "Loss (D-Real)": [], "Loss (D-Fake)": [], "Loss (D-GP)": []}

    initial_batch_count = 0
    start_time = time.time()

else:
    g_model, d_model = load_network()

    dict_loss = pd.read_csv(os.path.join(DATASET.name, S_LOGS, "loss.csv")).to_dict("list")
    initial_batch_count = int(dict_loss["Images trained"][-1] / BATCH_SIZE)
    start_time = time.time() - dict_loss["Time [s]"][-1]

last_status = time.time()
last_batch_count = initial_batch_count
last_status_loss = {"G": [], "D": [], "D-Real": [], "D-Fake": [], "D-GP": []}

for batch_count, (input_z, input_real) in enumerate(ds, start=initial_batch_count):

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        g_output = g_model(input_z, training=True)

        d_critics_real = d_model(input_real, training=True)
        d_critics_fake = d_model(g_output, training=True)

        g_loss = -tf.math.reduce_mean(d_critics_fake)

        d_loss_real = -tf.math.reduce_mean(d_critics_real)
        d_loss_fake = tf.math.reduce_mean(d_critics_fake)

        with tf.GradientTape() as gp_tape:
            alpha = tf.random.uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0.0, maxval=1.0)
            interpolated = alpha * input_real + (1 - alpha) * g_output
            gp_tape.watch(interpolated)
            d_critics_intp = d_model(interpolated)

        grads_intp = gp_tape.gradient(d_critics_intp, [interpolated])[0]

        grads_intp_l2 = tf.sqrt(tf.reduce_sum(tf.square(grads_intp), axis=[1, 2, 3]))
        grad_penalty = tf.reduce_mean(tf.square(tf.subtract(grads_intp_l2, 1)))

        d_loss_gp = tf.math.multiply(grad_penalty, LAMBDA_GP)
        d_loss = tf.math.accumulate_n([d_loss_real, d_loss_fake, d_loss_gp])

    d_grads = d_tape.gradient(d_loss, d_model.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, d_model.trainable_variables))

    g_grads = g_tape.gradient(g_loss, g_model.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, g_model.trainable_variables))

    last_status_loss["G"].append(g_loss.numpy())
    last_status_loss["D"].append(d_loss.numpy())
    last_status_loss["D-Real"].append(d_loss_real.numpy())
    last_status_loss["D-Fake"].append(d_loss_fake.numpy())
    last_status_loss["D-GP"].append(d_loss_gp.numpy())

    if time.time() - last_status > LOG_FREQUENCY:
        model_version += 1
        g_model.save(get_model_path(S_G))
        d_model.save(get_model_path(S_D))

        total_time = time.time() - start_time
        print_time = get_print_time(total_time)
        images_per_hour = BATCH_SIZE * (batch_count - last_batch_count) / (time.time() - last_status) * 3600

        dict_loss["Time [s]"].append(total_time)
        dict_loss["Loss (G)"].append(np.mean(last_status_loss["G"]))
        dict_loss["Loss (D)"].append(np.mean(last_status_loss["D"]))
        dict_loss["Loss (D-Real)"].append(np.mean(last_status_loss["D-Real"]))
        dict_loss["Loss (D-Fake)"].append(np.mean(last_status_loss["D-Fake"]))
        dict_loss["Loss (D-GP)"].append(np.mean(last_status_loss["D-GP"]))
        dict_loss["Images trained"].append(BATCH_SIZE * batch_count)
        dict_loss["Model version"].append(model_version)

        pd.DataFrame.from_dict(dict_loss).to_csv(os.path.join(DATASET.name, S_LOGS, "loss.csv"), index=False)

        plot_learning_curve()
        plot_dashboard()
        plot_generator_images()

        print(
            f"Version: {dict_loss['Model version'][-1]:4d} | Images trained: {dict_loss['Images trained'][-1]:8d} | "
            f"Time: {print_time.days}:{print_time.hours}:{print_time.minutes:02d}:{print_time.seconds:02d} | "
            f"G loss: {dict_loss['Loss (G)'][-1]:6.2f} | D loss: {dict_loss['Loss (D)'][-1]:6.2f} "
            f"(D-Real: {dict_loss['Loss (D-Real)'][-1]:6.2f}, D-Fake: {dict_loss['Loss (D-Fake)'][-1]:6.2f}, "
            f"D-GP: {dict_loss['Loss (D-GP)'][-1]:6.2f}) | Images per hour: {images_per_hour:6.0f}"
        )

        last_status = time.time()
        last_batch_count = batch_count
        last_status_loss = {"G": [], "D": [], "D-Real": [], "D-Fake": [], "D-GP": []}

        if dict_loss['Images trained'][-1] > N_TRAINING_IMAGES:
            break
