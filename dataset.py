import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from util import NNEncode, encode_313bin


class mydata(Dataset):
    def __init__(
        self,
        img_path,
        img_size,
        km_file_path,
        color_info,
        transform=None,
        NN=10.0,
        sigma=5.0,
    ):
        """
        Expected folder structure:

        img_path/
            SAR/
            OPT/

        Filenames must match.
        """

        self.img_size = img_size
        self.color_info = color_info

        self.sar_path = os.path.join(img_path, "SAR")
        self.opt_path = os.path.join(img_path, "OPT")

        self.img = sorted(os.listdir(self.sar_path))

        self.res_normalize_mean = [0.485, 0.456, 0.406]
        self.res_normalize_std = [0.229, 0.224, 0.225]

        if self.color_info == "dist":
            self.nnenc = NNEncode(NN, sigma, km_filepath=km_file_path)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        img_item = {}

        name = self.img[i]

        # -------------------------
        # 1. LOAD SAR (INPUT)
        # -------------------------
        sar_img = Image.open(
            os.path.join(self.sar_path, name)
        ).convert("L")  # SAR is grayscale

        # -------------------------
        # 2. LOAD OPTICAL (TARGET)
        # -------------------------
        rgb_image = Image.open(
            os.path.join(self.opt_path, name)
        ).convert("RGB")

        # resize
        sar_img = sar_img.resize((self.img_size, self.img_size), Image.LANCZOS)
        rgb_image = rgb_image.resize((self.img_size, self.img_size), Image.LANCZOS)

        sar_np = np.array(sar_img)
        rgb_np = np.array(rgb_image)

        # -------------------------
        # 3. LAB from optical
        # -------------------------
        lab_image = rgb2lab(rgb_np)
        l_image = sar_np[:, :, np.newaxis]  # <-- SAR replaces L channel
        ab_image = lab_image[:, :, 1:]

        # -------------------------
        # 4. color feature
        # -------------------------
        if self.color_info == "dist":
            color_feat = encode_313bin(
                np.expand_dims(ab_image, axis=0), self.nnenc
            )[0]
            color_feat = np.mean(color_feat, axis=(0, 1))

        else:
            color_feat = np.zeros(313)

        # -------------------------
        # 5. ResNet input (from SAR)
        # -------------------------
        gray_rgb = np.repeat(sar_np[:, :, np.newaxis], 3, axis=2) / 255.0
        res_input = (gray_rgb - self.res_normalize_mean) / self.res_normalize_std

        index = i + 0.0

        img_item["l_channel"] = np.transpose(l_image, (2, 0, 1)).astype(np.float32)
        img_item["ab_channel"] = np.transpose(ab_image, (2, 0, 1)).astype(np.float32)
        img_item["color_feat"] = color_feat.astype(np.float32)
        img_item["res_input"] = np.transpose(res_input, (2, 0, 1)).astype(np.float32)
        img_item["index"] = np.array(([index])).astype(np.float32)[0]

        return img_item