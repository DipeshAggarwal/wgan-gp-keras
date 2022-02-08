from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import ReLU

from tensorflow.keras.models import Sequential

def generator(latent_dim, weight_init, channels):
    model = Sequential(name="generator")
    
    model.add(Dense(8 * 8 * 512, input_dim=latent_dim))
    model.add(ReLU())
    
    model.add(Reshape((8, 8, 512)))
    model.add(Conv2DTranspose(256, (4, 4), (2, 2), padding="same", use_bias=False, kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(ReLU())
    
    model.add(Conv2DTranspose(128, (4, 4), (2, 2), padding="same", use_bias=False, kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(ReLU())
    
    model.add(Conv2DTranspose(64, (4, 4), (2, 2), padding="same", use_bias=False, kernel_initializer=weight_init))
    model.add(BatchNormalization())
    model.add(ReLU())
    
    model.add(Conv2D(channels, (4, 4), padding="same", activation="tanh"))
    
    return model

def critic(height, width, depth, alpha=0.2):
    model = Sequential(name="critic")
    input_shape = (height, width, depth)
    
    model.add(Conv2D(64, (4, 4), strides=(2, 2), padding="same", input_shape=input_shape))
    model.add(LeakyReLU(alpha=alpha))
    
    model.add(Conv2D(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=alpha))
    
    model.add(Conv2D(128, (4, 4), strides=(2, 2), padding="same"))
    model.add(LeakyReLU(alpha=alpha))
    
    model.add(Flatten())
    model.add(Dense(1, activation="linear"))
    
    return model
