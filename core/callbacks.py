from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import array_to_img
from matplotlib import pyplot as plt

import tensorflow as tf

class GANMonitor(Callback):
    
    def __init__(self, num_images=16, latent_dim=100):
        self.num_images = num_images
        self.latent_dim = latent_dim
        
        self.seed = tf.random.normal([num_images, latent_dim])
        
    def on_epoch_end(self, epoch, logs=None):
        generated_images = self.model.generator(self.seed)
        generated_images = (generated_images * 127.5) + 127.5
        generated_images.numpy()
        
        fig = plt.figure(figsize=(4, 4))
        for i in range(self.num_images):
            plt.subplot(4, 4, i+1)
            img = array_to_img(generated_images[i])
            plt.imshow(img)
            plt.axis("off")
        plt.savefig("output/epochs/epoch{:03d}.png".format(epoch))
        plt.show()
        
    def on_train_end(self, logs=None):
        self.model.generator.save("output/generator.h5")
