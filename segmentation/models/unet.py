from keras.layers import (
    BatchNormalization,
    Conv2D,
    UpSampling2D,
    ZeroPadding2D,
    concatenate,
)

from .backbones import default_encoder, resnet50, vgg16
from .model_utils import get_segmentation_model


def _unet(n_classes, encoder, input_height=256, input_width=256, channels=3):
    img_input, skip_connections = encoder(
        input_height=input_height, input_width=input_width, channels=channels
    )
    [skip_conn1, skip_conn2, skip_conn3, skip_conn4, skip_conn5] = skip_connections

    encoder_out = skip_conn4

    # block 1
    pad1 = (ZeroPadding2D((1, 1)))(encoder_out)
    conv1 = (Conv2D(512, (3, 3), padding="valid", activation="relu"))(pad1)
    norm1 = (BatchNormalization())(conv1)

    # block 2
    up2 = (UpSampling2D((2, 2)))(norm1)
    merge2 = concatenate([up2, skip_conn3])
    pad2 = (ZeroPadding2D((1, 1)))(merge2)
    conv2 = (Conv2D(256, (3, 3), padding="valid", activation="relu"))(pad2)
    norm2 = (BatchNormalization())(conv2)

    # block 3
    up3 = (UpSampling2D((2, 2)))(norm2)
    merge3 = concatenate([up3, skip_conn2])
    pad3 = (ZeroPadding2D((1, 1)))(merge3)
    conv3 = (Conv2D(128, (3, 3), padding="valid", activation="relu"))(pad3)
    norm3 = (BatchNormalization())(conv3)

    # block 4
    up4 = (UpSampling2D((2, 2)))(norm3)
    merge4 = concatenate([up4, skip_conn1])
    pad4 = (ZeroPadding2D((1, 1)))(merge4)
    conv4 = (Conv2D(64, (3, 3), padding="valid", activation="relu"))(pad4)
    norm4 = (BatchNormalization())(conv4)

    # final conv
    conv5 = Conv2D(n_classes, (3, 3), padding="same")(norm4)

    model = get_segmentation_model(img_input, conv5)

    return model


def unet(n_classes, input_height=256, input_width=256, channels=3):
    model = _unet(
        n_classes,
        default_encoder,
        input_height=input_height,
        input_width=input_width,
        channels=channels,
    )
    model.model_name = "unet"
    return model


def vgg16_unet(n_classes, input_height=256, input_width=256, channels=3):
    model = _unet(
        n_classes,
        vgg16,
        input_height=input_height,
        input_width=input_width,
        channels=channels,
    )
    model.model_name = "vgg16_unet"
    return model


def resnet50_unet(n_classes, input_height=256, input_width=256, channels=3):
    model = _unet(
        n_classes,
        resnet50,
        input_height=input_height,
        input_width=input_width,
        channels=channels,
    )
    model.model_name = "resnet50_unet"
    return model
