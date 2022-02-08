from core.loss import d_wasserstein_loss
from core.loss import g_wasserstein_loss
from core.nn.conv.wgan import generator
from core.nn.conv.wgan import critic
from core.callbacks import GANMonitor
from core.model import WGAN_GP

import tensorflow as tf
import numpy as np
import config

train_images = tf.keras.utils.image_dataset_from_directory(
    "dataset/images/", label_mode=None, image_size=(config.IMAGE_WIDTH, config.IMAGE_HEIGHT), batch_size=config.BATCH_SIZE
)
train_images = train_images.map(lambda x: (x - 127.5) / 127.5)

generator = generator(config.LATENT_DIM, tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02), channels=config.CHANNELS)
critic = critic(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, config.CHANNELS)

wgan = WGAN_GP(critic=critic, generator=generator, latent_dim=config.LATENT_DIM, critic_extra_steps=config.EXTRA_STEPS)

d_opt = tf.keras.optimizers.Adam(learning_rate=config.LR, beta_1=0.5, beta_2=0.9)
g_opt = tf.keras.optimizers.Adam(learning_rate=config.LR, beta_1=0.5, beta_2=0.9)

wgan.compile(
    d_optimiser=d_opt,
    g_optimiser=g_opt,
    d_loss_fn=d_wasserstein_loss,
    g_loss_fn=g_wasserstein_loss,
)

callback = [GANMonitor(num_images=16, latent_dim=config.LATENT_DIM)]
wgan.fit(train_images, epochs=config.EPOCHS, callbacks=callback)
