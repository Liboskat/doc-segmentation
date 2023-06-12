from keras.layers import BatchNormalization, Conv2D, UpSampling2D, ZeroPadding2D

from .backbones import default_encoder, resnet50, vgg16
from .model_utils import get_segmentation_model


def _segnet(n_classes, encoder, input_height=256, input_width=256, channels=3):
    img_input, block_outputs = encoder(
        input_height=input_height, input_width=input_width, channels=channels
    )
    encoder_out = block_outputs[3]

    # block 1
    pad1 = (ZeroPadding2D((1, 1)))(encoder_out)
    conv1 = (Conv2D(512, (3, 3), padding="valid"))(pad1)
    norm1 = (BatchNormalization())(conv1)

    # block 2
    up2 = (UpSampling2D((2, 2)))(norm1)
    pad2 = (ZeroPadding2D((1, 1)))(up2)
    conv2 = (Conv2D(256, (3, 3), padding="valid"))(pad2)
    norm2 = (BatchNormalization())(conv2)

    # block 3
    up3 = (UpSampling2D((2, 2)))(norm2)
    pad3 = (ZeroPadding2D((1, 1)))(up3)
    conv3 = (Conv2D(128, (3, 3), padding="valid"))(pad3)
    norm3 = (BatchNormalization())(conv3)

    # block 4
    up4 = (UpSampling2D((2, 2)))(norm3)
    pad4 = (ZeroPadding2D((1, 1)))(up4)
    conv4 = (Conv2D(64, (3, 3), padding="valid"))(pad4)
    norm4 = (BatchNormalization())(conv4)

    # block 5
    conv5 = Conv2D(n_classes, (3, 3), padding="same")(norm4)

    model = get_segmentation_model(img_input, conv5)

    return model


def segnet(n_classes, input_height=256, input_width=256, channels=3):
    model = _segnet(
        n_classes,
        default_encoder,
        input_height=input_height,
        input_width=input_width,
        channels=channels,
    )
    model.model_name = "segnet"
    return model


def vgg16_segnet(n_classes, input_height=256, input_width=256, channels=3):
    model = _segnet(
        n_classes,
        vgg16,
        input_height=input_height,
        input_width=input_width,
        channels=channels,
    )
    model.model_name = "vgg16_segnet"
    return model


def resnet50_segnet(n_classes, input_height=256, input_width=256, channels=3):
    model = _segnet(
        n_classes,
        resnet50,
        input_height=input_height,
        input_width=input_width,
        channels=channels,
    )
    model.model_name = "resnet50_segnet"
    return model
