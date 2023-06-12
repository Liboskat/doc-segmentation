# from segmentation.models.segnet import segnet
# from segmentation.models.unet import vgg16_unet
from segmentation.models.pspnet import resnet50_pspnet
from segmentation.predict import evaluate_segmentation, predict_segmentation

# train and check model
model = resnet50_pspnet(n_classes=4)
model.train(
    train_images="doc_dataset/img_train/",
    train_annotations="doc_dataset/anno_train/",
    checkpoints_path="./tmp/pspnet",
    epochs=3,
    steps_per_epoch=128,
)
metrics = model.evaluate_segmentation(
    inp_images_dir="doc_dataset/img_test",
    annotations_dir="doc_dataset/anno_test",
)
print(metrics)
model.predict_segmentation(
    inp="doc_dataset/img_test/3832349.jpg", out_fname="./tmp/out1.png"
)

# check for last epoch checkpoint
metrics = evaluate_segmentation(
    checkpoints_path="models\\pspnet\\resnet50_pspnet\\resnet50_pspnet",
    inp_images_dir="doc_dataset/img_test",
    annotations_dir="doc_dataset/anno_test",
)
predict_segmentation(
    checkpoints_path="models\\unet\\vgg16_unet\\vgg16_unet",
    inp="doc_dataset/img_test/3832349.jpg",
    out_fname="./tmp/tt.png",
    overlay_img=True,
    show_legends=True,
    class_names=["Background", "Passport", "Photo", "MRZ"],
)
