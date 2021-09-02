import imgaug as ia
import imgaug.augmenters as iaa

__all__ = ["seq"]

sometimes = lambda aug: iaa.Sometimes(0.1, aug)

seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.Flipud(0.5),  # vertically flip 20% of all images
        sometimes(
            iaa.CropAndPad(percent=(-0.05, 0.05), pad_mode=ia.ALL, pad_cval=(0, 255))
        ),
        sometimes(
            iaa.Affine(
                scale={
                    "x": (0.8, 1.2),
                    "y": (0.8, 1.2),
                },  # scale images to 80-120% of their size, individually per axis
                translate_percent={
                    "x": (-0.2, 0.2),
                    "y": (-0.2, 0.2),
                },  # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL,  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )
        ),
    ],
    random_order=True,
)
