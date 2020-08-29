import tensorflow as tf
import os

gen_model = tf.keras.models.load_model(os.path.join('objects', 'gen_model_004-001.h5'))
gen_model.summary()

gen_inputs = gen_model.input
gen_outputs = gen_model.layers[1](gen_inputs)
for i in range(2, len(gen_model.layers) - 1):
    gen_outputs = gen_model.layers[i](gen_outputs)

gen_outputs = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=5, padding='same', strides=2, use_bias=False, name='test1')(
    gen_outputs)
gen_outputs = tf.keras.layers.BatchNormalization(name='test2')(gen_outputs)
gen_outputs = tf.keras.layers.LeakyReLU(name='test3')(gen_outputs)
gen_outputs = tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=5, padding='same', use_bias=False,
                                              activation='tanh',name='test4')(gen_outputs)

new_gen_model = tf.keras.Model(inputs=gen_inputs, outputs=gen_outputs, name='gen_model')
new_gen_model.summary()

disc_model = tf.keras.models.load_model(os.path.join('objects', 'disc_model_004-001.h5'))
disc_model.summary()

disc_inputs = tf.keras.Input(shape=(8, 8, 3))
disc_outputs = tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', strides=1, name='hi1')(disc_inputs)
disc_outputs = tf.keras.layers.LayerNormalization(name='hello1')(disc_outputs)
disc_outputs = tf.keras.layers.LeakyReLU(name='world1')(disc_outputs)
disc_outputs = tf.keras.layers.Conv2D(filters=512, kernel_size=5, padding='same', strides=2, name='hi')(disc_outputs)
disc_outputs = disc_model.layers[2](disc_outputs)
for i in range(3, len(disc_model.layers)):
    disc_outputs = disc_model.layers[i](disc_outputs)

new_disc_model = tf.keras.Model(inputs=disc_inputs, outputs=disc_outputs, name='disc_model')
new_disc_model.summary()
