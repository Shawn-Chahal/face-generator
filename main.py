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
    ax.plot(dict_loss["Epoch"], dict_loss["Loss (G)"], label="G")
    ax.plot(dict_loss["Epoch"], dict_loss["Loss (D)"], label="D")
    # ax.plot(dict_loss["Epoch"], dict_loss["Loss (D-Real)"], label="D-Real")
    # ax.plot(dict_loss["Epoch"], dict_loss["Loss (D-Fake)"], label="D-Fake")
    # ax.plot(dict_loss["Epoch"], dict_loss["Loss (D-GP)"], label="D-GP")
    ax.legend(ncol=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(os.path.join(dataset, "logs", "learning_curve.png"))


def dashboard(width, height, n_last_epochs=None):
    n_rows = 1
    n_cols = 5

    if n_last_epochs is None:
        start = 0
    else:
        start = -min(n_last_epochs, epoch)

    fig = plt.figure(figsize=(width, height), dpi=600)

    ax = fig.add_subplot(n_rows, n_cols, 1)
    ax.plot(dict_loss["Epoch"][start:], dict_loss["Loss (G)"][start:], color="tab:blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (G)")

    ax = fig.add_subplot(n_rows, n_cols, 2)
    ax.plot(
        dict_loss["Epoch"][start:],
        dict_loss["Loss (D)"][start:],
        color="tab:orange",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (D)")

    ax = fig.add_subplot(n_rows, n_cols, 3)
    ax.plot(
        dict_loss["Epoch"][start:],
        dict_loss["Loss (D-Real)"][start:],
        color="tab:green",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (D-Real)")

    ax = fig.add_subplot(n_rows, n_cols, 4)
    ax.plot(
        dict_loss["Epoch"][start:], dict_loss["Loss (D-Fake)"][start:], color="tab:red"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (D-Fake)")

    ax = fig.add_subplot(n_rows, n_cols, 5)
    ax.plot(
        dict_loss["Epoch"][start:], dict_loss["Loss (D-GP)"][start:], color="tab:purple"
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (D-GP)")

    plt.tight_layout()

    if n_last_epochs is None:
        plt.savefig(os.path.join(dataset, "logs", "learning_curve_dashboard.png"))
    else:
        plt.savefig(
            os.path.join(dataset, "logs", "learning_curve_dashboard_recent.png")
        )


def get_time(t):
    hours = int(t / 3600)
    t = t - 3600 * hours
    minutes = int(t / 60)
    t = t - 60 * minutes
    seconds = int(t)

    return hours, minutes, seconds


def load_optimizers():
    g_optimizer = pickle.load(
        open(os.path.join(dataset, "objects", "g_optimizer.pkl"), "rb")
    )
    d_optimizer = pickle.load(
        open(os.path.join(dataset, "objects", "d_optimizer.pkl"), "rb")
    )

    return g_optimizer, d_optimizer


def load_network(dimensions, epoch):
    gen_model = tf.keras.models.load_model(
        os.path.join(dataset, "objects", f"gen_model_{dimensions:03d}-{epoch:03d}.h5")
    )
    disc_model = tf.keras.models.load_model(
        os.path.join(dataset, "objects", f"disc_model_{dimensions:03d}-{epoch:03d}.h5")
    )

    return gen_model, disc_model


def grow_network_from(dimensions, epoch):
    gen_model = tf.keras.models.load_model(
        os.path.join(dataset, "objects", f"gen_model_{dimensions:03d}-{epoch:03d}.h5")
    )
    gen_inputs = gen_model.input
    gen_outputs = gen_model.layers[1](gen_inputs)
    for i in range(2, len(gen_model.layers) - 1):
        gen_outputs = gen_model.layers[i](gen_outputs)

    gen_outputs = tf.keras.layers.Conv2DTranspose(
        filters=FILTERS[2 * dimensions],
        kernel_size=KERNEL_SIZE,
        padding="same",
        strides=2,
        use_bias=False,
        name=f"conv2d_transpose_{2 * dimensions:03d}",
    )(gen_outputs)

    gen_outputs = tf.keras.layers.BatchNormalization(
        name=f"batch_norm_{2 * dimensions:03d}"
    )(gen_outputs)
    gen_outputs = tf.keras.layers.LeakyReLU(name=f"leaky_relu_{2 * dimensions:03d}")(
        gen_outputs
    )
    gen_outputs = tf.keras.layers.Conv2DTranspose(
        filters=CHANNELS,
        kernel_size=KERNEL_SIZE,
        padding="same",
        use_bias=False,
        activation="tanh",
        name=f"output_{2 * dimensions:03d}",
    )(gen_outputs)

    new_gen_model = tf.keras.Model(
        inputs=gen_inputs, outputs=gen_outputs, name="gen_model"
    )

    disc_model = tf.keras.models.load_model(
        os.path.join(dataset, "objects", f"disc_model_{dimensions:03d}-{epoch:03d}.h5")
    )
    disc_inputs = tf.keras.Input(
        shape=((2 * dimensions), (2 * dimensions), CHANNELS), name="input"
    )
    disc_outputs = tf.keras.layers.Conv2D(
        filters=FILTERS[2 * dimensions],
        kernel_size=KERNEL_SIZE,
        padding="same",
        name=f"conv2d_{2 * dimensions:03d}",
    )(disc_inputs)

    disc_outputs = tf.keras.layers.LayerNormalization(
        name=f"layer_norm_{2 * dimensions:03d}"
    )(disc_outputs)

    disc_outputs = tf.keras.layers.LeakyReLU(name=f"leaky_relu_{2 * dimensions:03d}")(
        disc_outputs
    )

    disc_outputs = tf.keras.layers.Conv2D(
        filters=FILTERS[dimensions],
        kernel_size=KERNEL_SIZE,
        padding="same",
        strides=2,
        name=f"conv2d_{dimensions:03d}",
    )(disc_outputs)

    disc_outputs = disc_model.layers[2](disc_outputs)
    for i in range(3, len(disc_model.layers)):
        disc_outputs = disc_model.layers[i](disc_outputs)

    new_disc_model = tf.keras.Model(
        inputs=disc_inputs, outputs=disc_outputs, name="disc_model"
    )

    return new_gen_model, new_disc_model


def create_base_network():
    output_dim = 4

    gen_inputs = tf.keras.Input(shape=(Z_SIZE,), name="input")
    gen_hidden = tf.keras.layers.Reshape((1, 1, Z_SIZE), name="reshape")(gen_inputs)
    gen_hidden = tf.keras.layers.Conv2DTranspose(
        filters=FILTERS[output_dim],
        kernel_size=output_dim,
        use_bias=False,
        name=f"conv2d_transpose_{output_dim:03d}",
    )(gen_hidden)

    gen_hidden = tf.keras.layers.BatchNormalization(
        name=f"batch_norm_{output_dim:03d}"
    )(gen_hidden)
    gen_hidden = tf.keras.layers.LeakyReLU(name=f"leaky_relu_{output_dim:03d}")(
        gen_hidden
    )
    gen_outputs = tf.keras.layers.Conv2DTranspose(
        filters=CHANNELS,
        kernel_size=KERNEL_SIZE,
        padding="same",
        use_bias=False,
        activation="tanh",
        name=f"output_{output_dim:03d}",
    )(gen_hidden)

    disc_inputs = tf.keras.Input(shape=(output_dim, output_dim, CHANNELS), name="input")
    disc_hidden = tf.keras.layers.Conv2D(
        filters=FILTERS[output_dim],
        kernel_size=KERNEL_SIZE,
        padding="same",
        name=f"conv2d_{output_dim:03d}",
    )(disc_inputs)

    disc_hidden = tf.keras.layers.LayerNormalization(
        name=f"layer_norm_{output_dim:03d}"
    )(disc_hidden)
    disc_hidden = tf.keras.layers.LeakyReLU(name=f"leaky_relu_{output_dim:03d}")(
        disc_hidden
    )
    disc_hidden = tf.keras.layers.Conv2D(
        filters=FILTERS[output_dim], kernel_size=output_dim, name=f"conv2d_{1:03d}"
    )(disc_hidden)
    disc_hidden = tf.keras.layers.LayerNormalization(name=f"layer_norm_{1:03d}")(
        disc_hidden
    )
    disc_hidden = tf.keras.layers.LeakyReLU(name=f"leaky_relu_{1:03d}")(disc_hidden)
    disc_hidden = tf.keras.layers.Flatten(name="flatten")(disc_hidden)
    disc_outputs = tf.keras.layers.Dense(1, name="output")(disc_hidden)

    gen_model = tf.keras.Model(inputs=gen_inputs, outputs=gen_outputs, name="gen_model")
    disc_model = tf.keras.Model(
        inputs=disc_inputs, outputs=disc_outputs, name="disc_model"
    )

    return gen_model, disc_model


def process_path(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_image(img, channels=CHANNELS)
    img = tf.image.convert_image_dtype(img, tf.float32)

    if dataset is "celeba":
        img = tf.image.resize_with_crop_or_pad(img, 178, 178)

    img = tf.image.resize(img, size=(gen_dim, gen_dim))
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=NOISE)
    img = tf.image.random_contrast(img, lower=(1 / (1 + NOISE)), upper=(1 + NOISE))
    img = tf.image.random_saturation(img, lower=(1 / (1 + NOISE)), upper=(1 + NOISE))

    img = img * 2 - 1.0

    z_vector = tf.random.normal(shape=(Z_SIZE,))

    return z_vector, img


def int_log2(x):
    for i in range(16):
        if 2 ** i == x:
            return i


tf.random.set_seed(1)

image_dict = {
    "celeba": str(os.path.join("photos", "img_align_celeba", "*.jpg")),
    "flickr_faces": str(os.path.join("photos", "thumbnails128x128", "*", "*.png")),
}

dataset = "celeba"
initial_epoch = 11

NOISE = 0.1
CHANNELS = 3
KERNEL_SIZE = 5
Z_SIZE = 128
FILTERS = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}

LAMBDA_GP = 10
BETA_1 = 0
G_LR = 0.0001  # Generator learning rate
D_LR = 0.0004  # Discriminator learning rate

BUFFER_SIZE = 4096
BATCH_SIZE = 64
epochs_per_dim = 10
gen_dim_max = 128
gen_dim = min(2 ** (2 + int((initial_epoch - 1) / epochs_per_dim)), gen_dim_max)
last_growth_epoch = (int_log2(gen_dim_max) - 2) * epochs_per_dim + 1
STATUS_FREQUENCY = 60  # seconds
images_path = image_dict[dataset]
final_epoch = 1000

n_rows = 4
n_cols = 8
fixed_z = tf.random.normal(shape=(n_rows * n_cols, Z_SIZE))

list_ds = tf.data.Dataset.list_files(images_path)
n_images = len(list(list_ds))
list_ds.shuffle(buffer_size=n_images, reshuffle_each_iteration=False)
ds = list_ds.map(process_path)
ds = ds.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

n_batches = int(n_images / BATCH_SIZE)

if initial_epoch == 0:
    gen_model, disc_model = create_base_network()
    dict_loss = {
        "Epoch": [],
        "Loss (G)": [],
        "Loss (D)": [],
        "Loss (D-Real)": [],
        "Loss (D-Fake)": [],
        "Loss (D-GP)": [],
        "Dimensions": [],
        "Batch size": [],
        "Time [s]": [],
    }

    g_optimizer = tf.keras.optimizers.Adam(learning_rate=G_LR, beta_1=BETA_1)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=D_LR, beta_1=BETA_1)

else:
    gen_model, disc_model = load_network(gen_dim, initial_epoch)
    g_optimizer, d_optimizer = load_optimizers()
    dict_loss = pd.read_csv(os.path.join(dataset, "logs", "loss.csv")).to_dict("list")

with open(os.path.join(dataset, "logs", f"gen_model_summary_{gen_dim:03d}.txt"), "w") as f_gen_model_summary:
    gen_model.summary(print_fn=(lambda x: f_gen_model_summary.write(f"{x}\n")))

gen_model.summary()

with open(os.path.join(dataset, "logs", f"disc_model_summary_{gen_dim:03d}.txt"), "w") as f_disc_model_summary:
    disc_model.summary(print_fn=(lambda x: f_disc_model_summary.write(f"{x}\n")))

disc_model.summary()

start_time = time.time()
last_status = time.time()

for epoch in range(initial_epoch + 1, final_epoch + 1):

    gen_dim = min(2 ** (2 + int((epoch - 1) / epochs_per_dim)), gen_dim_max)

    if ((epoch - 1) % epochs_per_dim == 0) and (gen_dim <= gen_dim_max) and (
            epochs_per_dim < epoch <= last_growth_epoch):
        gen_model, disc_model = grow_network_from(gen_dim // 2, epoch - 1)
        ds = list_ds.map(process_path)
        ds = ds.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        with open(os.path.join(dataset, "logs", f"gen_model_summary_{gen_dim:03d}.txt"), "w") as f_gen_model_summary:
            gen_model.summary(print_fn=(lambda x: f_gen_model_summary.write(f"{x}\n")))

        with open(os.path.join(dataset, "logs", f"disc_model_summary_{gen_dim:03d}.txt"), "w") as f_disc_model_summary:
            disc_model.summary(print_fn=(lambda x: f_disc_model_summary.write(f"{x}\n")))

    epoch_loss = {"G": [], "D": [], "D-Real": [], "D-Fake": [], "D-GP": []}

    for i, (input_z, input_real) in enumerate(ds):

        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            g_output = gen_model(input_z, training=True)

            d_critics_real = disc_model(input_real, training=True)
            d_critics_fake = disc_model(g_output, training=True)

            g_loss = -tf.math.reduce_mean(d_critics_fake)

            d_loss_real = -tf.math.reduce_mean(d_critics_real)
            d_loss_fake = tf.math.reduce_mean(d_critics_fake)

            with tf.GradientTape() as gp_tape:
                alpha = tf.random.uniform(
                    shape=[d_critics_real.shape[0], 1, 1, 1], minval=0.0, maxval=1.0
                )
                interpolated = alpha * input_real + (1 - alpha) * g_output
                gp_tape.watch(interpolated)
                d_critics_intp = disc_model(interpolated)

            grads_intp = gp_tape.gradient(
                d_critics_intp,
                [
                    interpolated,
                ],
            )[0]

            grads_intp_l2 = tf.sqrt(
                tf.reduce_sum(tf.square(grads_intp), axis=[1, 2, 3])
            )
            grad_penalty = tf.reduce_mean(tf.square(grads_intp_l2 - 1.0))

            d_loss_gp = LAMBDA_GP * grad_penalty

            d_loss = d_loss_real + d_loss_fake + d_loss_gp

        d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
        d_optimizer.apply_gradients(
            grads_and_vars=zip(d_grads, disc_model.trainable_variables)
        )

        g_grads = g_tape.gradient(g_loss, gen_model.trainable_variables)
        g_optimizer.apply_gradients(
            grads_and_vars=zip(g_grads, gen_model.trainable_variables)
        )

        epoch_loss["G"].append(g_loss.numpy())
        epoch_loss["D"].append(d_loss.numpy())
        epoch_loss["D-Real"].append(d_loss_real.numpy())
        epoch_loss["D-Fake"].append(d_loss_fake.numpy())
        epoch_loss["D-GP"].append(d_loss_gp.numpy())

        if time.time() - last_status > STATUS_FREQUENCY:
            session_epochs = i / n_batches + epoch - initial_epoch - 1
            session_hours = (time.time() - start_time) / 3600
            epochs_per_hour = session_epochs / session_hours
            last_status = time.time()
            print(
                f"Epoch: {epoch} | Batch: {i:4d}/{n_batches} | Epochs per hour: {epochs_per_hour:.2f} | "
                f"Minutes per epoch: {60 / epochs_per_hour:.0f}"
            )

    dict_loss["Loss (G)"].append(np.mean(np.array(epoch_loss["G"])))
    dict_loss["Loss (D)"].append(np.mean(np.array(epoch_loss["D"])))
    dict_loss["Loss (D-Real)"].append(np.mean(np.array(epoch_loss["D-Real"])))
    dict_loss["Loss (D-Fake)"].append(np.mean(np.array(epoch_loss["D-Fake"])))
    dict_loss["Loss (D-GP)"].append(np.mean(np.array(epoch_loss["D-GP"])))
    dict_loss["Dimensions"].append(gen_dim)
    dict_loss["Batch size"].append(BATCH_SIZE)
    dict_loss["Epoch"].append(epoch)

    gen_model.save(
        os.path.join(
            dataset,
            "objects",
            f"gen_model_{gen_dim:03d}-{epoch:03d}.h5",
        )
    )

    disc_model.save(
        os.path.join(
            dataset,
            "objects",
            f"disc_model_{gen_dim:03d}-{epoch:03d}.h5",
        )
    )

    pickle.dump(
        g_optimizer,
        open(os.path.join(dataset, "objects", "g_optimizer.pkl"), "wb"),
    )

    pickle.dump(
        d_optimizer,
        open(os.path.join(dataset, "objects", "d_optimizer.pkl"), "wb"),
    )

    elapsed_time = time.time() - start_time
    elapsed_time_r = get_time(elapsed_time)

    if initial_epoch is 0:
        total_time = elapsed_time
    else:
        total_time = elapsed_time + dict_loss["Time [s]"][initial_epoch - 1]

    total_time_r = get_time(total_time)
    dict_loss["Time [s]"].append(total_time)
    pd.DataFrame.from_dict(dict_loss).to_csv(
        os.path.join(dataset, "logs", "loss.csv"), index=False
    )

    print(
        f"Epoch: {epoch} | "
        f"Time: {total_time_r[0]}:{total_time_r[1]:02d}:{total_time_r[2]:02d} | "
        f"G loss: {dict_loss['Loss (G)'][-1]:3.2f} | D loss: {dict_loss['Loss (D)'][-1]:3.2f} "
        f"(D-Real: {dict_loss['Loss (D-Real)'][-1]:3.2f}, D-Fake: {dict_loss['Loss (D-Fake)'][-1]:3.2f}, "
        f"D-GP: {dict_loss['Loss (D-GP)'][-1]:3.2f})"
    )

    learning_curve(6, 4)
    dashboard(20, 4)

    epoch_samples = (gen_model(fixed_z, training=False).numpy() + 1) / 2

    fig = plt.figure(figsize=(n_cols, n_rows), dpi=300)
    for i in range(n_rows * n_cols):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(epoch_samples[i])

    plt.tight_layout()
    plt.savefig(os.path.join(dataset, "logs", f"generated_faces_{gen_dim:03d}-{epoch:03d}.png"))
    plt.savefig(os.path.join(dataset, "logs", f"generated_faces_latest.png"))
