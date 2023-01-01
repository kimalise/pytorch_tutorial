import cv2
import albumentations as A
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import plot_examples

image = Image.open("albumentations/images/dog1.jpg")
# image.show()
width, height = image.size
print(width, height)

transforms = A.Compose([
    A.Resize(width=width // 2, height=height // 2),
    A.RandomCrop(height=height // 4, width=width // 4),
    A.Rotate(limit=40, p=0.9, border_mode=cv2.BORDER_CONSTANT),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=0.9),
    A.OneOf(
        [
            A.Blur(blur_limit=3, p=0.5),
            A.ColorJitter(p=0.5),
        ],
        p=1.0,
    ),
])

images_list = [image]
image = np.array(image)

for i in tqdm(range(15)):
    augmentations = transforms(image=image)
    augmented_img = augmentations["image"]
    images_list.append(augmented_img)

plot_examples(images_list)



