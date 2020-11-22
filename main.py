import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import os
import pickle
import time


def learning_curve(width, height):
    fig = plt.figure(figsize=(width, height), dpi=600)

    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dict_loss["Images trained"], dict_loss["Loss (G)"], label="G")
    ax.plot(dict_loss["Images trained"], dict_loss["Loss (D)"], label="D")
    ax.legend()
    ax.set_xlabel("Images trained")
    ax.set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(dataset, "logs", "learning_curve.png"))
    plt.close(fig)


def dashboard(width, height):
    n_rows = 1
    n_cols = 5

    fig = plt.figure(figsize=(width, height), dpi=600)

    labels = ["Loss (G)", "Loss (D)", "Loss (D-Real)", "Loss (D-Fake)", "Loss (D-GP)"]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    for i, (label, color) in enumerate(zip(labels, colors), 1):
        ax = fig.add_subplot(n_rows, n_cols, i)
        ax.plot(dict_loss["Images trained"], dict_loss[label], color=color)
        ax.set_xlabel("Images trained")
        ax.set_ylabel(label)

    plt.tight_layout()
    plt.savefig(os.path.join(dataset, "logs", "learning_curve_dashboard.png"))
    plt.close(fig)


def plot_images(gen_images):
    image_batch = (gen_images.numpy() + 1) / 2
    fig = plt.figure(figsize=(n_cols, n_rows), dpi=300)
    for i in range(n_rows * n_cols):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image_batch[i])

    plt.tight_layout()
    plt.savefig(os.path.join(dataset, "logs", f"generated_faces_{model_version:04d}.png"))
    plt.savefig(os.path.join(dataset, "logs", f"generated_faces_latest.png"))
    plt.close(fig)


def get_time(t):
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    t = t - 60 * minutes
    seconds = int(t)

    return hours, minutes, seconds


def load_optimizers():
    g_optimizer = pickle.load(open(os.path.join(dataset, "objects", "g_optimizer.pkl"), "rb"))
    d_optimizer = pickle.load(open(os.path.join(dataset, "objects", "d_optimizer.pkl"), "rb"))

    return g_optimizer, d_optimizer


def load_network():
    gen_model = tf.keras.models.load_model(os.path.join(dataset, "objects", f"gen_model-{model_version:04d}.h5"))
    disc_model = tf.keras.models.load_model(os.path.join(dataset, "objects", f"disc_model-{model_version:04d}.h5"))

    return gen_model, disc_model


def create_network():
    min_dim = 4
    output_dim = min_dim + 0

    gen_rgb = []

    gen_inputs = tf.keras.Input(shape=(Z_SIZE,))
    gen_hidden = tf.keras.layers.Reshape((1, 1, Z_SIZE))(gen_inputs)

    gen_hidden = tf.keras.layers.Conv2DTranspose(filters=FILTERS[output_dim], kernel_size=output_dim, use_bias=False)(
        gen_hidden)
    gen_hidden = tf.keras.layers.BatchNormalization()(gen_hidden)
    gen_hidden = tf.keras.layers.LeakyReLU()(gen_hidden)

    gen_rgb.append(
        tf.keras.layers.Conv2DTranspose(filters=CHANNELS, kernel_size=KERNEL_SIZE, padding="same",
                                        use_bias=False)(gen_hidden))

    while output_dim < GEN_DIM:
        output_dim *= 2
        gen_hidden = tf.keras.layers.Conv2DTranspose(filters=FILTERS[output_dim], kernel_size=KERNEL_SIZE,
                                                     padding="same", use_bias=False, strides=2)(gen_hidden)

        gen_hidden = tf.keras.layers.BatchNormalization()(gen_hidden)
        gen_hidden = tf.keras.layers.LeakyReLU()(gen_hidden)
        gen_rgb.append(tf.keras.layers.Conv2DTranspose(filters=CHANNELS, kernel_size=KERNEL_SIZE, padding="same",
                                                       use_bias=False)(gen_hidden))

    gen_outputs = gen_rgb[0]

    for gen_rgb_skip in gen_rgb[1:]:
        gen_outputs = tf.keras.layers.UpSampling2D(size=2, interpolation='bilinear')(gen_outputs)
        gen_outputs = tf.keras.layers.Add()([gen_outputs, gen_rgb_skip])

    gen_outputs = tf.math.tanh(gen_outputs)

    gen_model = tf.keras.Model(inputs=gen_inputs, outputs=gen_outputs)

    disc_inputs = tf.keras.Input(shape=(output_dim, output_dim, CHANNELS))
    disc_skip = disc_inputs
    disc_outputs = tf.keras.layers.Conv2D(filters=FILTERS[output_dim], kernel_size=KERNEL_SIZE, padding="same"
                                          )(disc_inputs)
    disc_outputs = tf.keras.layers.LayerNormalization()(disc_outputs)
    disc_outputs = tf.keras.layers.LeakyReLU()(disc_outputs)

    while output_dim > min_dim:
        output_dim = output_dim // 2

        disc_outputs = tf.keras.layers.Conv2D(filters=FILTERS[output_dim], kernel_size=KERNEL_SIZE, padding="same",
                                              strides=2)(disc_outputs)

        disc_skip = tf.keras.layers.AveragePooling2D(pool_size=2, padding='same')(disc_skip)
        disc_frgb = tf.keras.layers.Conv2D(filters=FILTERS[output_dim], kernel_size=KERNEL_SIZE, padding="same"
                                           )(disc_skip)

        disc_outputs = tf.keras.layers.Average()([disc_outputs, disc_frgb])
        disc_outputs = tf.keras.layers.LayerNormalization()(disc_outputs)
        disc_outputs = tf.keras.layers.LeakyReLU()(disc_outputs)

    disc_outputs = tf.keras.layers.Conv2D(filters=Z_SIZE, kernel_size=output_dim)(disc_outputs)
    disc_outputs = tf.keras.layers.LayerNormalization()(disc_outputs)
    disc_outputs = tf.keras.layers.LeakyReLU()(disc_outputs)
    disc_outputs = tf.keras.layers.Flatten()(disc_outputs)
    disc_outputs = tf.keras.layers.Dense(1)(disc_outputs)

    disc_model = tf.keras.Model(inputs=disc_inputs, outputs=disc_outputs)

    with open(os.path.join(dataset, "logs", f"gen_model_summary.txt"), "w") as f_gen_model_summary:
        gen_model.summary(print_fn=(lambda x: f_gen_model_summary.write(f"{x}\n")))

    gen_model.summary()

    with open(os.path.join(dataset, "logs", f"disc_model_summary.txt"), "w") as f_disc_model_summary:
        disc_model.summary(print_fn=(lambda x: f_disc_model_summary.write(f"{x}\n")))

    disc_model.summary()

    return gen_model, disc_model


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if dataset is "celeba":
        img = tf.image.resize_with_crop_or_pad(img, 178, 178)

    img = tf.image.resize(img, size=(GEN_DIM, GEN_DIM))
    img = tf.image.random_flip_left_right(img)

    img = img * 2 - 1.0

    z_vector = tf.random.normal(shape=(Z_SIZE,))

    return z_vector, img


tf.random.set_seed(1)

image_dict = {
    "celeba": str(os.path.join("photos", "img_align_celeba", "*.jpg")),
    "flickr_faces": str(os.path.join("photos", "thumbnails128x128", "*", "*.png")),
}

dataset = "celeba"
model_version = 144
log_frequency = 12 * 60  # seconds

GEN_DIM = 64
BATCH_SIZE = 32

Z_SIZE = 128
FILTERS = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 32}
CHANNELS = 3
KERNEL_SIZE = 5
BUFFER_SIZE = 4096

LAMBDA_GP = 10
BETA_1 = 0
G_LR = 0.0001  # Generator learning rate
D_LR = 0.0004  # Discriminator learning rate

images_path = image_dict[dataset]

n_rows = 4
n_cols = 8
fixed_z = tf.random.normal(shape=(n_rows * n_cols, Z_SIZE))

list_ds = tf.data.Dataset.list_files(images_path)
n_images = len(list(list_ds))
list_ds.shuffle(buffer_size=n_images, reshuffle_each_iteration=False)
ds = list_ds.map(process_path)
ds = ds.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()

if model_version == 0:
    gen_model, disc_model = create_network()
    dict_loss = {"Model version": [], "Images trained": [], "Time [s]": [], "Loss (G)": [], "Loss (D)": [],
                 "Loss (D-Real)": [], "Loss (D-Fake)": [], "Loss (D-GP)": []}

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=G_LR, beta_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LR, beta_1=BETA_1)

    initial_batch_count = 0
    start_time = time.time()

else:
    gen_model, disc_model = load_network()
    g_optimizer, d_optimizer = load_optimizers()
    dict_loss = pd.read_csv(os.path.join(dataset, "logs", "loss.csv")).to_dict("list")
    initial_batch_count = int(dict_loss["Images trained"][-1] / BATCH_SIZE)
    start_time = time.time() - dict_loss["Time [s]"][-1]

last_status = time.time()
last_batch_count = initial_batch_count
last_status_loss = {"G": [], "D": [], "D-Real": [], "D-Fake": [], "D-GP": []}

for batch_count, (input_z, input_real) in enumerate(ds, initial_batch_count):

    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        g_output = gen_model(input_z, training=True)

        d_critics_real = disc_model(input_real, training=True)
        d_critics_fake = disc_model(g_output, training=True)

        g_loss = -tf.math.reduce_mean(d_critics_fake)

        d_loss_real = -tf.math.reduce_mean(d_critics_real)
        d_loss_fake = tf.math.reduce_mean(d_critics_fake)

        with tf.GradientTape() as gp_tape:
            alpha = tf.random.uniform(shape=[d_critics_real.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)
            interpolated = alpha * input_real + (1 - alpha) * g_output
            gp_tape.watch(interpolated)
            d_critics_intp = disc_model(interpolated)

        grads_intp = gp_tape.gradient(d_critics_intp, [interpolated])[0]

        grads_intp_l2 = tf.sqrt(tf.reduce_sum(tf.square(grads_intp), axis=[1, 2, 3]))
        grad_penalty = tf.reduce_mean(tf.square(grads_intp_l2 - 1.0))

        d_loss_gp = LAMBDA_GP * grad_penalty
        d_loss = d_loss_real + d_loss_fake + d_loss_gp

    d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
    d_optimizer.apply_gradients(zip(d_grads, disc_model.trainable_variables))

    g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)
    g_optimizer.apply_gradients(zip(g_grads, gen_model.trainable_variables))

    last_status_loss["G"].append(g_loss.numpy())
    last_status_loss["D"].append(d_loss.numpy())
    last_status_loss["D-Real"].append(d_loss_real.numpy())
    last_status_loss["D-Fake"].append(d_loss_fake.numpy())
    last_status_loss["D-GP"].append(d_loss_gp.numpy())

    if time.time() - last_status > log_frequency:
        model_version += 1
        gen_model.save(os.path.join(dataset, "objects", f"gen_model-{model_version:04d}.h5"))
        disc_model.save(os.path.join(dataset, "objects", f"disc_model-{model_version:04d}.h5"))
        pickle.dump(g_optimizer, open(os.path.join(dataset, "objects", "g_optimizer.pkl"), "wb"))
        pickle.dump(d_optimizer, open(os.path.join(dataset, "objects", "d_optimizer.pkl"), "wb"))

        total_time = time.time() - start_time
        total_time_r = get_time(total_time)
        images_per_hour = BATCH_SIZE * (batch_count - last_batch_count) / (time.time() - last_status) * 3600

        dict_loss["Time [s]"].append(total_time)
        dict_loss["Loss (G)"].append(np.mean(last_status_loss["G"]))
        dict_loss["Loss (D)"].append(np.mean(last_status_loss["D"]))
        dict_loss["Loss (D-Real)"].append(np.mean(last_status_loss["D-Real"]))
        dict_loss["Loss (D-Fake)"].append(np.mean(last_status_loss["D-Fake"]))
        dict_loss["Loss (D-GP)"].append(np.mean(last_status_loss["D-GP"]))
        dict_loss["Images trained"].append(BATCH_SIZE * batch_count)
        dict_loss["Model version"].append(model_version)

        pd.DataFrame.from_dict(dict_loss).to_csv(os.path.join(dataset, "logs", "loss.csv"), index=False)

        learning_curve(6, 4)
        dashboard(20, 4)
        plot_images(gen_model(fixed_z))

        print(
            f"Version: {dict_loss['Model version'][-1]} | Images trained: {dict_loss['Images trained'][-1]} | "
            f"Time: {total_time_r[0]}:{total_time_r[1]:02d}:{total_time_r[2]:02d} | "
            f"G loss: {dict_loss['Loss (G)'][-1]:3.2f} | D loss: {dict_loss['Loss (D)'][-1]:3.2f} "
            f"(D-Real: {dict_loss['Loss (D-Real)'][-1]:3.2f}, D-Fake: {dict_loss['Loss (D-Fake)'][-1]:3.2f}, "
            f"D-GP: {dict_loss['Loss (D-GP)'][-1]:3.2f}) | Images per hour: {images_per_hour:.0f}"
        )

        last_status = time.time()
        last_batch_count = batch_count
        last_status_loss = {"G": [], "D": [], "D-Real": [], "D-Fake": [], "D-GP": []}
