"""A U-NET model for segmentation of atomic force microscopy image grains."""

from keras.models import Model
from keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    concatenate,
    Conv2DTranspose,
    Dropout,
)
from keras.optimizers import Adam

# Disable pylint line too long since this is much more readable with each layer on its own line
# pylint: disable=line-too-long

# Disable pylint too many variables since this is the nature of the U-NET model
# pylint: disable=too-many-locals


def unet_model(img_height: int, img_width: int, img_channels: int, learning_rate: float = 0.001):
    """U-NET model definition function."""

    inputs = Input((img_height, img_width, img_channels))

    # Downsampling
    # Downsample with increasing numbers of filters to try to capture more complex features (first argument)
    # Dropout is used to try to prevent overfitting. Increase if overfitting happens.
    # Dropout increases deeper into the model to further help prevent overfitting.

    conv1 = Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(inputs)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv1)
    pooled1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(pooled1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv2)
    pooled2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(pooled2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv3)
    pooled3 = MaxPooling2D((2, 2))(conv3)

    conv4 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(pooled3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv4)
    pooled4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(pooled4)
    conv5 = Dropout(0.3)(conv5)
    conv5 = Conv2D(256, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv5)

    # Upsampling
    # Conv2DTranspose is used as a sort of inverse convolution, to upsample the image
    # A concatenation is used to force context from the original image, providing information about what context a
    # feature stems from.

    up6 = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv5)
    up6 = concatenate([up6, conv4])
    conv6 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(up6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv6)

    up7 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv6)
    up7 = concatenate([up7, conv3])
    conv7 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(up7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv7)

    up8 = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv7)
    up8 = concatenate([up8, conv2])
    conv8 = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(up8)
    conv8 = Dropout(0.1)(conv8)
    conv8 = Conv2D(32, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv8)

    up9 = Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding="same")(conv8)
    up9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(up9)
    conv9 = Dropout(0.1)(conv9)
    conv9 = Conv2D(16, kernel_size=(3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(conv9)

    # Output layer
    # Sigmoid activation function to force output to be between 0 and 1
    outputs = Conv2D(1, kernel_size=(1, 1), activation="sigmoid")(conv9)

    optimiser = Adam(lr=learning_rate)

    # Compile the model
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimiser, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()

    return model
