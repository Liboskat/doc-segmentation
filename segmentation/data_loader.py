import itertools
import os
import random

import cv2
import numpy as np
import six

random.seed(1338)
class_colors = [
    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    for _ in range(5000)
]

ACCEPTABLE_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp"]
ACCEPTABLE_SEGMENTATION_FORMATS = [".png", ".bmp"]


class DataLoaderError(Exception):
    pass


def get_image_list_from_path(images_path):
    image_files = []
    for dir_entry in os.listdir(images_path):
        if (
            os.path.isfile(os.path.join(images_path, dir_entry))
            and os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS
        ):
            image_files.append(os.path.join(images_path, dir_entry))
    return image_files


def get_pairs_from_paths(images_path, annotations_path, ignore_non_matching=False):
    image_files = []
    segmentation_files = {}

    for dir_entry in os.listdir(images_path):
        if (
            os.path.isfile(os.path.join(images_path, dir_entry))
            and os.path.splitext(dir_entry)[1] in ACCEPTABLE_IMAGE_FORMATS
        ):
            file_name, file_extension = os.path.splitext(dir_entry)
            image_files.append(
                (file_name, file_extension, os.path.join(images_path, dir_entry))
            )

    for dir_entry in os.listdir(annotations_path):
        if (
            os.path.isfile(os.path.join(annotations_path, dir_entry))
            and os.path.splitext(dir_entry)[1] in ACCEPTABLE_SEGMENTATION_FORMATS
        ):
            file_name, file_extension = os.path.splitext(dir_entry)
            full_dir_entry = os.path.join(annotations_path, dir_entry)
            if file_name in segmentation_files:
                raise DataLoaderError(
                    "Segmentation file with filename {0}"
                    " already exists and is ambiguous to"
                    " resolve with path {1}."
                    " Please remove or rename the latter.".format(
                        file_name, full_dir_entry
                    )
                )

            segmentation_files[file_name] = (file_extension, full_dir_entry)

    return_value = []
    # Match the images and segmentations
    for image_file, _, image_full_path in image_files:
        if image_file in segmentation_files:
            return_value.append((image_full_path, segmentation_files[image_file][1]))
        elif ignore_non_matching:
            continue
        else:
            # Error out
            raise DataLoaderError(
                "No corresponding segmentation "
                "found for image {0}.".format(image_full_path)
            )

    return return_value


def get_image_array(image_input, width, height):
    if type(image_input) is np.ndarray:
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError(
                "get_image_array: path {0} doesn't exist".format(image_input)
            )
        img = cv2.imread(image_input, cv2.IMREAD_COLOR)
    else:
        raise DataLoaderError(
            "get_image_array: Can't process input type {0}".format(
                str(type(image_input))
            )
        )

    # sub_mean
    img = cv2.resize(img, (width, height))
    img = img.astype(np.float32)
    img = np.atleast_3d(img)

    means = [103.939, 116.779, 123.68]

    for i in range(min(img.shape[2], len(means))):
        img[:, :, i] -= means[i]
    return img[:, :, ::-1]


def get_segmentation_array(image_input, nClasses, width, height, no_reshape=False):
    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError(
                "get_segmentation_array: " "path {0} doesn't exist".format(image_input)
            )
        img = cv2.imread(image_input, cv2.IMREAD_COLOR)
    else:
        raise DataLoaderError(
            "get_segmentation_array: "
            "Can't process input type {0}".format(str(type(image_input)))
        )

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width * height, nClasses))

    return seg_labels


def verify_segmentation_dataset(images_path, annotations_path, n_classes):
    try:
        img_seg_pairs = get_pairs_from_paths(images_path, annotations_path)
        if not len(img_seg_pairs):
            print(
                "Couldn't load any data from images_path: "
                "{0} and segmentations path: {1}".format(images_path, annotations_path)
            )
            return False

        return_value = True
        for im_fn, seg_fn in iter(img_seg_pairs):
            img = cv2.imread(im_fn)
            seg = cv2.imread(seg_fn)
            # Check dimensions match
            if not img.shape == seg.shape:
                return_value = False
                print(
                    "The size of image {0} and its segmentation {1} "
                    "doesn't match (possibly the files are corrupt).".format(
                        im_fn, seg_fn
                    )
                )
            else:
                max_pixel_value = np.max(seg[:, :, 0])
                if max_pixel_value >= n_classes:
                    return_value = False
                    print(
                        "The pixel values of the segmentation image {0} "
                        "violating range [0, {1}]. "
                        "Found maximum pixel value {2}".format(
                            seg_fn, str(n_classes - 1), max_pixel_value
                        )
                    )
        if return_value:
            print("Dataset verified! ")
        else:
            print("Dataset not verified!")
        return return_value
    except DataLoaderError as e:
        print("Found error during data loading\n{0}".format(str(e)))
        return False


def image_segmentation_generator(
    images_path,
    segs_path,
    n_classes,
    input_height,
    input_width,
    output_height,
    output_width,
    batch_size=2,
):
    img_seg_pairs = get_pairs_from_paths(images_path, segs_path)
    random.shuffle(img_seg_pairs)
    zipped = itertools.cycle(img_seg_pairs)

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            im, seg = next(zipped)
            im = cv2.imread(im, cv2.IMREAD_COLOR)
            seg = cv2.imread(seg, cv2.IMREAD_COLOR)

            X.append(get_image_array(im, input_width, input_height))
            Y.append(
                get_segmentation_array(seg, n_classes, output_width, output_height)
            )

            yield np.array(X), np.array(Y)
