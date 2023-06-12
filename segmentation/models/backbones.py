from keras import Input
from keras.applications import VGG16, ResNet50
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D,
)


def default_encoder(input_height=256, input_width=256, channels=3):
    img_input = Input(shape=(input_height, input_width, channels))
    block_outputs = []

    # block 1
    pad1 = (ZeroPadding2D((1, 1)))(img_input)
    conv1 = (Conv2D(64, (3, 3), padding="valid"))(pad1)
    norm1 = (BatchNormalization())(conv1)
    act1 = (Activation("relu"))(norm1)
    pool1 = (MaxPooling2D((2, 2)))(act1)
    block_outputs.append(pool1)

    # block 2
    pad2 = (ZeroPadding2D((1, 1)))(pool1)
    conv2 = (Conv2D(128, (3, 3), padding="valid"))(pad2)
    norm2 = (BatchNormalization())(conv2)
    act2 = (Activation("relu"))(norm2)
    pool2 = (MaxPooling2D((2, 2)))(act2)
    block_outputs.append(pool2)

    # blocks 3-5
    pool = pool2
    for _ in range(3):
        pad = (ZeroPadding2D((1, 1)))(pool)
        conv = (Conv2D(256, (3, 3), padding="valid"))(pad)
        norm = (BatchNormalization())(conv)
        act = (Activation("relu"))(norm)
        pool = (MaxPooling2D((2, 2)))(act)
        block_outputs.append(pool)

    return img_input, block_outputs


def vgg16(input_height=224, input_width=224, channels=3):
    vgg = VGG16(input_shape=(input_height, input_width, channels), include_top=False)
    block_out1 = vgg.get_layer(name="block1_pool").output
    block_out2 = vgg.get_layer(name="block2_pool").output
    block_out3 = vgg.get_layer(name="block3_pool").output
    block_out4 = vgg.get_layer(name="block4_pool").output
    block_out5 = vgg.get_layer(name="block5_pool").output
    return vgg.input, [block_out1, block_out2, block_out3, block_out4, block_out5]


def resnet50(input_height=224, input_width=224, channels=3):
    resnet = ResNet50(
        input_shape=(input_height, input_width, channels), include_top=False
    )
    block_out1 = resnet.get_layer(name="conv1_conv").output
    block_out2 = resnet.get_layer(name="conv2_block3_out").output
    block_out3 = resnet.get_layer(name="conv3_block4_out").output
    block_out4 = resnet.get_layer(name="conv4_block6_out").output
    block_out5 = resnet.get_layer(name="conv5_block3_out").output
    return resnet.input, [block_out1, block_out2, block_out3, block_out4, block_out5]
