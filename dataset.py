import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from skimage.color import rgb2lab
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
        Folder structure:

        img_path/
            SAR/
            OPT/
        """

        self.img_size = img_size
        self.color_info = color_info

        self.sar_path = os.path.join(img_path, "SAR")
        self.opt_path = os.path.join(img_path, "OPT")

        self.img = sorted(os.listdir(self.sar_path))

        # normalization for ResNet input
        self.res_mean = np.array([0.485, 0.456, 0.406])
        self.res_std = np.array([0.229, 0.224, 0.225])

        if self.color_info == "dist":
            self.nnenc = NNEncode(NN, sigma, km_filepath=km_file_path)

    def __len__(self):
        return len(self.img)

    def __getitem__(self, i):
        name = self.img[i]

        # -------------------------
        # 1. LOAD SAR (INPUT)
        # -------------------------
        sar = Image.open(os.path.join(self.sar_path, name)).convert("L")
        sar = sar.resize((self.img_size, self.img_size), Image.LANCZOS)
        sar_np = np.array(sar).astype(np.float32)

        # normalize SAR to [0,1]
        sar_norm = sar_np / 255.0

        # -------------------------
        # 2. LOAD OPTICAL (TARGET)
        # -------------------------
        rgb = Image.open(os.path.join(self.opt_path, name)).convert("RGB")
        rgb = rgb.resize((self.img_size, self.img_size), Image.LANCZOS)
        rgb_np = np.array(rgb).astype(np.float32)

        # -------------------------
        # 3. LAB from optical
        # -------------------------
        lab = rgb2lab(rgb_np / 255.0)

        # IMPORTANT: scale SAR → LAB L range
        l = sar_norm * 100.0  # now in [0,100]
        l = l[:, :, np.newaxis]

        ab = lab[:, :, 1:]

        # -------------------------
        # 4. color feature
        # -------------------------
        if self.color_info == "dist":
            color_feat = encode_313bin(
                np.expand_dims(ab, axis=0), self.nnenc
            )[0]
            color_feat = np.mean(color_feat, axis=(0, 1))
        else:
            color_feat = np.zeros(313, dtype=np.float32)

        # -------------------------
        # 5. ResNet input (from SAR)
        # -------------------------
        gray_rgb = np.repeat(sar_norm[:, :, np.newaxis], 3, axis=2)
        res_input = (gray_rgb - self.res_mean) / self.res_std

        return {
            "l_channel": np.transpose(l, (2, 0, 1)).astype(np.float32),
            "ab_channel": np.transpose(ab, (2, 0, 1)).astype(np.float32),
            "color_feat": color_feat.astype(np.float32),
            "res_input": np.transpose(res_input, (2, 0, 1)).astype(np.float32),
            "index": float(i),
        }