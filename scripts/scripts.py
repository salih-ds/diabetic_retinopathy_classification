import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

def crop_image1(img,tol=7):
    # img is image data
    # tol  is tolerance

    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def crop_image(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        h,w,_=img.shape
#         print(h,w)
        img1=cv2.resize(crop_image1(img[:,:,0]),(w,h))
        img2=cv2.resize(crop_image1(img[:,:,1]),(w,h))
        img3=cv2.resize(crop_image1(img[:,:,2]),(w,h))

#         print(img1.shape,img2.shape,img3.shape)
        img[:,:,0]=img1
        img[:,:,1]=img2
        img[:,:,2]=img3

        # conver to image tensor
        # res = T.ToImage()(img)
        res = img
        return res

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, crop=None, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img_name = self.img_labels.iloc[idx, 0]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # crop black borders
        if self.crop:
          image = crop_image(image)

        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            # image = self.transform(image)
            image = self.transform(image=image)["image"]
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_name

def transforms():
    transforms = A.Compose([
        A.Resize(380, 380),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return transforms

def augmentation():
    augmentation = A.Compose([
        A.Resize(380, 380),
        A.RandomSizedCrop(min_max_height=(int(380 * 0.75), 380),
                            height=380,
                            width=380, p=0.5),

        A.ElasticTransform(p=0.5, alpha_affine=20, border_mode=0),
        A.Rotate(p=0.5, border_mode=0, limit=45),
        A.HueSaturationValue(10,15,10,p=0.5),
        A.CLAHE(p=0.5),
        A.Flip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.5),
        A.Blur(blur_limit=5, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return augmentation