from types import MethodType

import keras.backend as K
from keras.layers import Activation, Lambda, Reshape
from keras.models import Model

from ..predict import evaluate_segmentation, predict_segmentation
from ..train import train


def resize_image(inp, s, data_format="channels_last"):
    try:
        return Lambda(
            lambda x: K.resize_images(
                x,
                height_factor=s[0],
                width_factor=s[1],
                data_format=data_format,
                interpolation="bilinear",
            )
        )(inp)
    except Exception:
        assert data_format == "channels_last"
        import tensorflow as tf

        return Lambda(
            lambda x: tf.image.resize_images(
                x, (K.int_shape(x)[1] * s[0], K.int_shape(x)[2] * s[1])
            )
        )(inp)


def get_segmentation_model(input, output):
    img_input = input
    o = output

    o_shape = Model(img_input, o).output_shape
    i_shape = Model(img_input, o).input_shape

    output_height = o_shape[1]
    output_width = o_shape[2]
    input_height = i_shape[1]
    input_width = i_shape[2]
    n_classes = o_shape[3]
    o = (Reshape((output_height * output_width, -1)))(o)

    o = (Activation("softmax"))(o)
    model = Model(img_input, o)
    model.output_width = output_width
    model.output_height = output_height
    model.n_classes = n_classes
    model.input_height = input_height
    model.input_width = input_width
    model.model_name = ""

    model.train = MethodType(train, model)
    model.predict_segmentation = MethodType(predict_segmentation, model)
    model.evaluate_segmentation = MethodType(evaluate_segmentation, model)

    return model
