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
    img = tf.io.decode_image(img, channels=CHANNELS, dtype=tf.float32, expand_animations=False)

    if DATASET.name is dataset_info.celeba.name:
        img = tf.image.resize_with_crop_or_pad(img, 178, 178)

    img = tf.image.resize(img, size=(GEN_DIM, GEN_DIM))
    img = tf.image.random_flip_left_right(img)
    img = tf.math.multiply(img, 2)
    img = tf.math.subtract(img, 1)

    return img


def create_network():
    def summarize_model():
        with open(os.path.join(DATASET.name, S_ENCODER, S_LOGS, f"e_model_summary.txt"), "w") as f_e_model_summary:
            nn_e_model.summary(print_fn=(lambda x: f_e_model_summary.write(f"{x}\n")))

        nn_e_model.summary()

    def basic_block(block_input, kernel_size, padding="same"):
        block_output = tf.keras.layers.Conv2D(filters=FILTERS[output_dim],
                                              kernel_size=kernel_size,
                                              padding=padding)(block_input)
        block_output = tf.keras.layers.LayerNormalization()(block_output)
        block_output = tf.keras.layers.LeakyReLU()(block_output)
        return block_output

    min_dim = 4
    output_dim = GEN_DIM + 0

    e_input = tf.keras.Input(shape=(output_dim, output_dim, CHANNELS))
    e_input_skip = e_input
    e_outputs = basic_block(e_input, kernel_size=KERNEL_SIZE_RGB)

    while output_dim > min_dim:
        output_dim = output_dim // 2

        e_outputs = tf.keras.layers.Conv2D(filters=FILTERS[output_dim], kernel_size=KERNEL_SIZE, padding="same",
                                           strides=2)(e_outputs)

        e_input_skip = tf.keras.layers.AveragePooling2D(padding='same')(e_input_skip)
        e_from_rgb = tf.keras.layers.Conv2D(filters=FILTERS[output_dim], kernel_size=KERNEL_SIZE_RGB, padding="same",
                                            strides=1)(e_input_skip)

        e_outputs = tf.keras.layers.Add()([e_outputs, e_from_rgb])
        e_outputs = tf.keras.layers.LayerNormalization()(e_outputs)
        e_outputs = tf.keras.layers.LeakyReLU()(e_outputs)

    e_outputs = tf.keras.layers.Conv2D(filters=FILTERS[output_dim],
                                       kernel_size=output_dim,
                                       padding="valid")(e_outputs)
    e_outputs = tf.keras.layers.Flatten()(e_outputs)

    nn_e_model = tf.keras.Model(inputs=e_input, outputs=e_outputs)

    summarize_model()

    return nn_e_model


def get_model_path():
    return os.path.join(DATASET.name, S_ENCODER, S_OBJECTS, f"e_model-{model_version:04d}.h5")


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
    ax.plot(dict_loss["Images trained"], dict_loss["Loss"], label="Encoder")
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_xlabel("Images trained")
    ax.set_ylabel("Loss")

    fig.savefig(os.path.join(DATASET.name, S_ENCODER, S_LOGS, "learning_curve.png"))
    plt.close(fig)


def plot_images():
    e_vector = e_model(FIXED_IMAGES)
    gen_images = g_model(e_vector)

    gen_image_batch = (gen_images.numpy() + 1) / 2
    fixed_image_batch = (FIXED_IMAGES.numpy() + 1) / 2

    fig = plt.figure(figsize=(N_COLS, N_ROWS), dpi=300, constrained_layout=True)
    for i in range(BATCH_SIZE):
        ax = fig.add_subplot(N_ROWS, N_COLS, 2 * i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(fixed_image_batch[i])
        ax.text(x=0.5, y=-0.1, s="Original", size=10, horizontalalignment='center', verticalalignment='top',
                transform=ax.transAxes)

        ax = fig.add_subplot(N_ROWS, N_COLS, 2 * i + 2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(gen_image_batch[i])
        ax.text(x=0.5, y=-0.1, s="Generator", size=10, horizontalalignment='center', verticalalignment='top',
                transform=ax.transAxes)

    fig.savefig(
        os.path.join(DATASET.name, S_ENCODER, S_ENCODER_IMAGES_LOCAL, f"encoded_images_{model_version:04d}.png"))
    fig.savefig(os.path.join(DATASET.name, S_ENCODER, S_LOGS, f"encoded_images_latest.png"))

    if model_version % LOG_FREQUENCY_GIT == 0:
        fig.savefig(
            os.path.join(DATASET.name, S_ENCODER, S_ENCODER_IMAGES, f"encoded_images_{model_version:04d}.png"))

    plt.close(fig)


def mean_squared_error(y_model, y_actual):
    mse = tf.math.subtract(y_model, y_actual)
    mse = tf.square(mse)
    mse = tf.reduce_mean(mse)
    return mse


tf.random.set_seed(1)

ReadableTime = namedtuple('ReadableTime', ['days', 'hours', 'minutes', 'seconds'])

S_ENCODER = "encoder"
S_LOGS = "logs"
S_OBJECTS = "objects"
S_ENCODER_IMAGES_LOCAL = "encoder_images_local"
S_ENCODER_IMAGES = "encoder_images"

LOG_FREQUENCY = 12 * 60  # seconds
LOG_FREQUENCY_GIT = 20  # versions

BUFFER_SIZE = 4096
BATCH_SIZE = 16
GEN_DIM = 128
CHANNELS = 3
KERNEL_SIZE = 5
KERNEL_SIZE_RGB = 1
FILTERS = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 32}
Z_SIZE = 512

N_IMAGES = 2 * BATCH_SIZE
N_COLS = 8
N_ROWS = int(np.ceil(N_IMAGES / N_COLS))

""" TRAINING_FUNCTION PARAMETERS """

DATASET = dataset_info.celeba
G_MODEL_VERSION = 205

model_version = 0

""" TRAINING_FUNCTION PARAMETERS """

g_model = tf.keras.models.load_model(os.path.join(DATASET.name, S_OBJECTS, f"g_model-{G_MODEL_VERSION:04d}.h5"))

ds_verify = tf.data.Dataset.list_files(DATASET.glob).map(process_image_path).batch(BATCH_SIZE, drop_remainder=True)

for batch in ds_verify:
    FIXED_IMAGES = batch
    break

list_ds = tf.data.Dataset.list_files(DATASET.glob)
list_ds = list_ds.shuffle(buffer_size=len(list(list_ds)), reshuffle_each_iteration=False)
ds = list_ds.map(process_image_path)
ds = ds.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()

optimizer = tf.keras.optimizers.Adam()  # learning_rate=LEARNING_RATE, beta_1=BETA_1)

if model_version == 0:
    e_model = create_network()

    dict_loss = {"Model version": [], "Images trained": [], "Time [s]": [], "Loss": []}

    initial_batch_count = 0
    start_time = time.time()

else:
    e_model = tf.keras.models.load_model(get_model_path())

    dict_loss = pd.read_csv(os.path.join(DATASET.name, S_ENCODER, S_LOGS, "loss.csv")).to_dict("list")
    initial_batch_count = int(dict_loss["Images trained"][-1] / BATCH_SIZE)
    start_time = time.time() - dict_loss["Time [s]"][-1]

last_status = time.time()
last_batch_count = initial_batch_count
last_status_loss = []

for batch_count, images_real in enumerate(ds, start=initial_batch_count):

    with tf.GradientTape() as tape:

        latent_z = e_model(images_real, training=True)
        images_generated = g_model(latent_z)
        loss = mean_squared_error(images_generated, images_real)

    grads = tape.gradient(loss, e_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, e_model.trainable_variables))

    last_status_loss.append(loss.numpy())

    if time.time() - last_status > LOG_FREQUENCY:
        model_version += 1
        e_model.save(get_model_path())

        total_time = time.time() - start_time
        print_time = get_print_time(total_time)
        images_per_hour = BATCH_SIZE * (batch_count - last_batch_count) / (time.time() - last_status) * 3600

        dict_loss["Time [s]"].append(total_time)
        dict_loss["Loss"].append(np.mean(last_status_loss))
        dict_loss["Images trained"].append(BATCH_SIZE * batch_count)
        dict_loss["Model version"].append(model_version)

        pd.DataFrame.from_dict(dict_loss).to_csv(os.path.join(DATASET.name, S_ENCODER, S_LOGS, "loss.csv"), index=False)

        plot_learning_curve()
        plot_images()

        uint8_loss = int(255 * (np.sqrt(dict_loss['Loss'][-1]) / 2))

        if len(dict_loss["Loss"]) > 1:
            delta_loss = dict_loss['Loss'][-1] - dict_loss['Loss'][-2]
            delta_time = (time.time() - last_status)
            etr = -dict_loss['Loss'][-1] / delta_loss * delta_time
            print_etr = get_print_time(etr)
        else:
            print_etr = ReadableTime(99, 23, 59, 59)

        print(
            f"Version: {dict_loss['Model version'][-1]:4d} | Images trained: {dict_loss['Images trained'][-1]:8d} | "
            f"Time: {print_time.days}:{print_time.hours}:{print_time.minutes:02d}:{print_time.seconds:02d} | "
            f"Loss: {dict_loss['Loss'][-1]:8.5f} | uint8 loss: {uint8_loss:03d} | Images per hour: {images_per_hour:6.0f} | "
            f"Estimated time remaining: {print_etr.days}:{print_etr.hours}:{print_etr.minutes:02d}:{print_etr.seconds:02d}"
        )

        last_status = time.time()
        last_batch_count = batch_count
        last_status_loss = []
