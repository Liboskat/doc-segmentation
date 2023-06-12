import json
import os

from keras.callbacks import ModelCheckpoint

from segmentation.data_loader import (
    image_segmentation_generator,
    verify_segmentation_dataset,
)


def train(
    model,
    train_images,
    train_annotations,
    verify_dataset=True,
    checkpoints_path=None,
    epochs=5,
    steps_per_epoch=512,
):
    n_classes = model.n_classes
    input_height = model.input_height
    input_width = model.input_width
    output_height = model.output_height
    output_width = model.output_width

    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    callbacks = []
    if checkpoints_path is not None:
        config_file = checkpoints_path + "_config.json"
        dir_name = os.path.dirname(config_file)

        if (not os.path.exists(dir_name)) and len(dir_name) > 0:
            os.makedirs(dir_name)

        with open(config_file, "w") as f:
            json.dump(
                {
                    "model_class": model.model_name,
                    "n_classes": n_classes,
                    "input_height": input_height,
                    "input_width": input_width,
                    "output_height": output_height,
                    "output_width": output_width,
                },
                f,
            )

        default_callback = ModelCheckpoint(
            filepath=checkpoints_path + ".{epoch:04d}",
            save_weights_only=True,
            verbose=True,
        )
        callbacks = [default_callback]

    if verify_dataset:
        print("Verifying training dataset")
        verified = verify_segmentation_dataset(
            train_images, train_annotations, n_classes
        )
        assert verified

    train_gen = image_segmentation_generator(
        train_images,
        train_annotations,
        n_classes,
        input_height,
        input_width,
        output_height,
        output_width,
    )

    model.fit(
        train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks
    )
