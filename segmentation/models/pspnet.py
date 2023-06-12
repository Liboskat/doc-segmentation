import keras.backend as K
import numpy as np
from keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    UpSampling2D,
)

from .backbones import default_encoder, resnet50, vgg16
from .model_utils import get_segmentation_model


def pool_block(encoder_output, pool_factor):
    h = K.int_shape(encoder_output)[1]
    w = K.int_shape(encoder_output)[2]

    pool_size = strides = [
        int(np.round(float(h) / pool_factor)),
        int(np.round(float(w) / pool_factor)),
    ]

    pool = AveragePooling2D(pool_size, strides=strides, padding="same")(encoder_output)
    conv = Conv2D(512, (1, 1), padding="same", use_bias=False)(pool)
    norm = BatchNormalization()(conv)
    act = Activation("relu")(norm)

    up = UpSampling2D(strides, interpolation="bilinear")(act)
    return up


def _pspnet(n_classes, encoder, input_height=384, input_width=384, channels=3):
    assert input_height % 192 == 0
    assert input_width % 192 == 0

    img_input, levels = encoder(
        input_height=input_height, input_width=input_width, channels=channels
    )
    [o1, o2, o3, o4, encoder_output] = levels

    pool_factors = [1, 2, 3, 6]
    pool_outs = [encoder_output]

    for pf in pool_factors:
        pooled = pool_block(encoder_output, pf)
        pool_outs.append(pooled)

    merge = Concatenate()(pool_outs)

    conv = Conv2D(512, (1, 1), use_bias=False)(merge)
    norm = BatchNormalization()(conv)
    act = Activation("relu")(norm)

    conv = Conv2D(n_classes, (3, 3), padding="same")(act)
    up = UpSampling2D((8, 8), interpolation="bilinear")(conv)

    model = get_segmentation_model(img_input, up)
    return model


def pspnet(n_classes, input_height=384, input_width=384, channels=3):
    model = _pspnet(
        n_classes,
        default_encoder,
        input_height=input_height,
        input_width=input_width,
        channels=channels,
    )
    model.model_name = "pspnet"
    return model


def vgg16_pspnet(n_classes, input_height=384, input_width=384, channels=3):
    model = _pspnet(
        n_classes,
        vgg16,
        input_height=input_height,
        input_width=input_width,
        channels=channels,
    )
    model.model_name = "vgg16_pspnet"
    return model


def resnet50_pspnet(n_classes, input_height=384, input_width=384, channels=3):
    model = _pspnet(
        n_classes,
        resnet50,
        input_height=input_height,
        input_width=input_width,
        channels=channels,
    )
    model.model_name = "resnet50_pspnet"
    return model
