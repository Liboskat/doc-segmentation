from . import pspnet, segnet, unet

model_from_name = {
    "pspnet": pspnet.pspnet,
    "vgg16_pspnet": pspnet.vgg16_pspnet,
    "resnet50_pspnet": pspnet.resnet50_pspnet,
    "unet": unet.unet,
    "vgg16_unet": unet.vgg16_unet,
    "resnet50_unet": unet.resnet50_unet,
    "segnet": segnet.segnet,
    "vgg16_segnet": segnet.vgg16_segnet,
    "resnet50_segnet": segnet.resnet50_segnet,
}
