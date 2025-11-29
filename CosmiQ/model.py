"""Main module to instantiate deepCR models and use them."""

from os import path, mkdir
import math
import shutil
import secrets

import numpy as np
import torch
import torch.nn as nn
from torch import from_numpy
from joblib import Parallel, delayed
from joblib import dump, load
from joblib import wrap_non_picklable_objects

from unet import WrappedModel, UNet2Sigmoid, UNet2SigmoidVI
from util import medmask

__all__ = ["deepCR", "deepCRVI"]


class deepCR:
    def __init__(self, mask, inpaint=None, device="GPU", hidden=32):
        """
        Simplified deepCR:

        Parameters
        ----------
        mask : str
            Full file path of your model (including '.pth').
        inpaint : (optional) str or None
            Currently unused; inpainting is median-based by default.
        device : str
            'CPU' or 'GPU'.
        hidden : int
            Number of hidden channels for first deepCR-mask layer.
        """

        # ----------------- Device setup -----------------
        if device.upper() == "GPU":
            if not torch.cuda.is_available():
                raise AssertionError("No CUDA device detected!")
            self.dtype = torch.cuda.FloatTensor
            self.dint = torch.cuda.ByteTensor
            wrapper = nn.DataParallel
        else:
            self.dtype = torch.FloatTensor
            self.dint = torch.ByteTensor
            wrapper = WrappedModel

        # ----------------- Mask network -----------------
        # Always treat `mask` as a full path to a .pth file
        self.scale = 1.0
        mask_path = mask

        self.maskNet = wrapper(UNet2Sigmoid(1, 1, hidden))
        self.maskNet.type(self.dtype)

        if device.upper() != "GPU":
            state = torch.load(mask_path, map_location="cpu")
        else:
            state = torch.load(mask_path)

        self.maskNet.load_state_dict(state)
        self.maskNet.eval()
        for p in self.maskNet.parameters():
            p.requires_grad = False

        # ----------------- Inpainting network -----------------
        # For now we always use median-based inpainting
        self.inpaintNet = None

        # ----------------- Normalization flags -----------------
        self.norm = True
        self.percentile = None
        self.median = None
        self.std = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clean(self, img0, threshold=0.5, inpaint=False, binary=True,
              segment=True, patch=1024, n_jobs=1):
        """
        Identify cosmic rays in an input image, and (optionally) inpaint
        with the predicted cosmic ray mask.

        Parameters
        ----------
        img0 : np.ndarray
            2D input image (e.g. HST ACS/WFC _flc.fits in electrons).
        threshold : float
            Threshold in [0, 1] for binarizing the probabilistic mask.
        inpaint : bool
            If True, also return a clean inpainted image.
        binary : bool
            If True, return binary CR mask; otherwise, probabilistic mask.
        segment : bool
            If True, segment into patches for memory control.
        patch : int
            Patch size if segment == True.
        n_jobs : int
            Parallel jobs for large images.

        Returns
        -------
        mask or (mask, inpainted)
        """

        img0 = img0.astype(np.float32) / self.scale
        img0 = img0.copy()
        if self.norm:
            self.median = img0.mean()
            self.std = img0.std()
            img0 -= self.median
            img0 /= self.std

        if not segment and n_jobs == 1:
            return self.clean_(img0, threshold=threshold,
                               inpaint=inpaint, binary=binary)
        else:
            if n_jobs == 1:
                return self.clean_large(img0, threshold=threshold,
                                        inpaint=inpaint, binary=binary,
                                        patch=patch)
            else:
                return self.clean_large_parallel(
                    img0, threshold=threshold,
                    inpaint=inpaint, binary=binary,
                    patch=patch, n_jobs=n_jobs
                )

    def clean_(self, img0, threshold=0.5, inpaint=True, binary=True):
        """
        Given input image, return cosmic ray mask and (optionally) clean image.
        """

        # pad to be divisible by 4
        shape = img0.shape
        pad_x = -shape[0] % 4
        pad_y = -shape[1] % 4
        img0 = np.pad(img0, ((pad_x, 0), (pad_y, 0)), mode="constant")

        img0_t = from_numpy(img0).type(self.dtype).view(1, 1, *img0.shape)
        mask = self.maskNet(img0_t)
        binary_mask = (mask > threshold).type(self.dtype)

        if binary:
            return_mask = (
                binary_mask.detach().squeeze().cpu().numpy()[pad_x:, pad_y:]
            )
        else:
            return_mask = (
                mask.detach().squeeze().cpu().numpy()[pad_x:, pad_y:]
            )

        if not inpaint:
            return return_mask

        # Inpainting
        if self.inpaintNet is not None:
            cat = torch.cat((img0_t * (1 - binary_mask), binary_mask), dim=1)
            img1 = self.inpaintNet(cat).detach()
            inpainted = img1 * binary_mask + img0_t * (1 - binary_mask)
            inpainted = inpainted.detach().cpu().squeeze().numpy()
        else:
            binary_mask_np = binary_mask.detach().cpu().squeeze().numpy()
            img0_np = img0_t.detach().cpu().squeeze().numpy()
            img1 = medmask(img0_np, binary_mask_np)
            inpainted = img1 * binary_mask_np + img0_np * (1 - binary_mask_np)

        if self.norm:
            inpainted *= self.std
            inpainted += self.median

        return return_mask, inpainted[pad_x:, pad_y:] * self.scale

    def clean_large_parallel(self, img0, threshold=0.5, inpaint=True,
                             binary=True, patch=256, n_jobs=-1):
        """
        Large-image version using joblib for parallel patch processing.
        """

        folder = "./joblib_memmap_" + secrets.token_hex(3)
        try:
            mkdir(folder)
        except FileExistsError:
            folder = "./joblib_memmap_" + secrets.token_hex(3)
            mkdir(folder)

        im_shape = img0.shape
        img0_dtype = img0.dtype
        hh = int(math.ceil(im_shape[0] / patch))
        ww = int(math.ceil(im_shape[1] / patch))

        img0_filename_memmap = path.join(folder, "img0_memmap")
        dump(img0, img0_filename_memmap)
        img0 = load(img0_filename_memmap, mmap_mode="r")

        if inpaint:
            img1_filename_memmap = path.join(folder, "img1_memmap")
            img1 = np.memmap(
                img1_filename_memmap, dtype=img0.dtype,
                shape=im_shape, mode="w+"
            )
        else:
            img1 = None

        mask_filename_memmap = path.join(folder, "mask_memmap")
        mask = np.memmap(
            mask_filename_memmap,
            dtype=np.int8 if binary else img0_dtype,
            shape=im_shape, mode="w+",
        )

        @wrap_non_picklable_objects
        def fill_values(i, j, img0, img1, mask, patch, inpaint, threshold, binary):
            img = img0[
                i * patch : min((i + 1) * patch, im_shape[0]),
                j * patch : min((j + 1) * patch, im_shape[1]),
            ]
            if inpaint:
                mask_, clean_ = self.clean_(img, threshold=threshold,
                                            inpaint=True, binary=binary)
                mask[
                    i * patch : min((i + 1) * patch, im_shape[0]),
                    j * patch : min((j + 1) * patch, im_shape[1]),
                ] = mask_
                img1[
                    i * patch : min((i + 1) * patch, im_shape[0]),
                    j * patch : min((j + 1) * patch, im_shape[1]),
                ] = clean_
            else:
                mask_ = self.clean_(img, threshold=threshold,
                                    inpaint=False, binary=binary)
                mask[
                    i * patch : min((i + 1) * patch, im_shape[0]),
                    j * patch : min((j + 1) * patch, im_shape[1]),
                ] = mask_

        _ = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(fill_values)(
                i, j, img0, img1, mask, patch, inpaint, threshold, binary
            )
            for i in range(hh)
            for j in range(ww)
        )

        mask = np.array(mask)
        if inpaint:
            img1 = np.array(img1)
        try:
            shutil.rmtree(folder)
        except Exception:
            print("Could not clean-up automatically.")

        if inpaint:
            return mask, img1
        else:
            return mask

    def clean_large(self, img0, threshold=0.5, inpaint=True, binary=True,
                    patch=256):
        """
        Serial large-image version (no joblib).
        """

        im_shape = img0.shape
        hh = int(math.ceil(im_shape[0] / patch))
        ww = int(math.ceil(im_shape[1] / patch))

        img1 = np.zeros((im_shape[0], im_shape[1]))
        mask = np.zeros((im_shape[0], im_shape[1]))

        if inpaint:
            for i in range(hh):
                for j in range(ww):
                    img = img0[
                        i * patch : min((i + 1) * patch, im_shape[0]),
                        j * patch : min((j + 1) * patch, im_shape[1]),
                    ]
                    mask_, clean_ = self.clean_(img, threshold=threshold,
                                                inpaint=True, binary=binary)
                    mask[
                        i * patch : min((i + 1) * patch, im_shape[0]),
                        j * patch : min((j + 1) * patch, im_shape[1]),
                    ] = mask_
                    img1[
                        i * patch : min((i + 1) * patch, im_shape[0]),
                        j * patch : min((j + 1) * patch, im_shape[1]),
                    ] = clean_
            return mask, img1

        else:
            for i in range(hh):
                for j in range(ww):
                    img = img0[
                        i * patch : min((i + 1) * patch, im_shape[0]),
                        j * patch : min((j + 1) * patch, im_shape[1]),
                    ]
                    mask_ = self.clean_(img, threshold=threshold,
                                        inpaint=False, binary=binary)
                    mask[
                        i * patch : min((i + 1) * patch, im_shape[0]),
                        j * patch : min((j + 1) * patch, im_shape[1]),
                    ] = mask_
            return mask

    def inpaint(self, img0, mask):
        """
        Inpaint img0 under mask.
        """
        img0 = img0.astype(np.float32) / self.scale
        mask = mask.astype(np.float32)
        shape = img0.shape[-2:]

        if self.inpaintNet is not None:
            img0_t = from_numpy(img0).type(self.dtype).view(
                1, -1, shape[0], shape[1]
            )
            mask_t = from_numpy(mask).type(self.dtype).view(
                1, -1, shape[0], shape[1]
            )
            cat = torch.cat((img0_t * (1 - mask_t), mask_t), dim=1)
            img1 = self.inpaintNet(cat)
            img1 = img1.detach()
            inpainted = img1 * mask_t + img0_t * (1 - mask_t)
            inpainted = (
                inpainted.detach().cpu().view(shape[0], shape[1]).numpy()
            )
        else:
            img1 = medmask(img0, mask)
            inpainted = img1 * mask + img0 * (1 - mask)

        return inpainted * self.scale


class deepCRVI:
    """
    Variational deepCR-like wrapper.
    - Uses UNet2SigmoidVI(1,1,hidden)
    - `mask` is a path to a .pth file
    - clean(img0, ...) works like deepCR.clean()
    """

    def __init__(self, mask, inpaint=None, device="GPU", hidden=32, n_samples=16):

        if device.upper() == "GPU":
            if not torch.cuda.is_available():
                raise AssertionError("No CUDA device detected!")
            self.dtype = torch.cuda.FloatTensor
            self.dint = torch.cuda.ByteTensor
            wrapper = nn.DataParallel
        else:
            self.dtype = torch.FloatTensor
            self.dint = torch.ByteTensor
            wrapper = WrappedModel

        self.scale = 1.0
        mask_path = mask
        self.maskNet = wrapper(UNet2SigmoidVI(1, 1, hidden))
        self.maskNet.type(self.dtype)

        if device.upper() != "GPU":
            state = torch.load(mask_path, map_location="cpu")
        else:
            state = torch.load(mask_path)
        self.maskNet.load_state_dict(state)

        self.maskNet.eval()
        for p in self.maskNet.parameters():
            p.requires_grad = False

        self.inpaintNet = None  # optional: can be added later
        self.norm = True
        self.percentile = None
        self.median = None
        self.std = None
        self.n_samples = n_samples

    # ---- same API as deepCR ----
    def clean(self, img0, threshold=0.5, inpaint=False, binary=True,
              segment=True, patch=1024, n_jobs=1):

        img0 = img0.astype(np.float32) / self.scale
        img0 = img0.copy()
        if self.norm:
            self.median = img0.mean()
            self.std = img0.std()
            img0 -= self.median
            img0 /= self.std

        if not segment and n_jobs == 1:
            return self.clean_(img0, threshold=threshold,
                               inpaint=inpaint, binary=binary)
        else:
            if n_jobs == 1:
                return self.clean_large(img0, threshold=threshold,
                                        inpaint=inpaint, binary=binary,
                                        patch=patch)
            else:
                return self.clean_large_parallel(
                    img0, threshold=threshold,
                    inpaint=inpaint, binary=binary,
                    patch=patch, n_jobs=n_jobs
                )

    def clean_(self, img0, threshold=0.5, inpaint=True, binary=True):

        shape = img0.shape
        pad_x = -shape[0] % 4
        pad_y = -shape[1] % 4
        img0 = np.pad(img0, ((pad_x, 0), (pad_y, 0)), mode="constant")

        img0_t = from_numpy(img0).type(self.dtype).view(1, 1, *img0.shape)

        # Monte Carlo sampling over VI weights
        with torch.no_grad():
            probs_list = []
            for _ in range(self.n_samples):
                p = self.maskNet(img0_t)
                probs_list.append(p)
            probs_stack = torch.stack(probs_list, dim=0)   # [T,1,1,H,W]
            mask = probs_stack.mean(dim=0)                 # [1,1,H,W]

            # predictive entropy (store for later inspection)
            p_mean = mask.clamp(1e-6, 1 - 1e-6)
            entropy = -(p_mean * p_mean.log() +
                        (1 - p_mean) * (1 - p_mean).log())
            self.last_entropy = (
                entropy.detach().cpu().squeeze().numpy()[pad_x:, pad_y:]
            )

        binary_mask = (mask > threshold).type(self.dtype)
        if binary:
            return_mask = (
                binary_mask.detach().squeeze().cpu().numpy()[pad_x:, pad_y:]
            )
        else:
            return_mask = (
                mask.detach().squeeze().cpu().numpy()[pad_x:, pad_y:]
            )

        if not inpaint:
            return return_mask

        # basic median inpainting
        binary_mask_np = binary_mask.detach().cpu().squeeze().numpy()
        img0_np = img0_t.detach().cpu().squeeze().numpy()
        img1 = medmask(img0_np, binary_mask_np)
        inpainted = img1 * binary_mask_np + img0_np * (1 - binary_mask_np)

        if self.norm:
            inpainted *= self.std
            inpainted += self.median

        return return_mask, inpainted[pad_x:, pad_y:] * self.scale

    def clean_large_parallel(self, img0, threshold=0.5, inpaint=True,
                             binary=True, patch=256, n_jobs=-1):

        folder = "./joblib_memmap_" + secrets.token_hex(3)
        try:
            mkdir(folder)
        except FileExistsError:
            folder = "./joblib_memmap_" + secrets.token_hex(3)
            mkdir(folder)

        im_shape = img0.shape
        img0_dtype = img0.dtype
        hh = int(math.ceil(im_shape[0] / patch))
        ww = int(math.ceil(im_shape[1] / patch))

        img0_filename_memmap = path.join(folder, "img0_memmap")
        dump(img0, img0_filename_memmap)
        img0 = load(img0_filename_memmap, mmap_mode="r")

        if inpaint:
            img1_filename_memmap = path.join(folder, "img1_memmap")
            img1 = np.memmap(
                img1_filename_memmap, dtype=img0.dtype,
                shape=im_shape, mode="w+"
            )
        else:
            img1 = None

        mask_filename_memmap = path.join(folder, "mask_memmap")
        mask = np.memmap(
            mask_filename_memmap,
            dtype=np.int8 if binary else img0_dtype,
            shape=im_shape, mode="w+",
        )

        @wrap_non_picklable_objects
        def fill_values(i, j, img0, img1, mask, patch, inpaint, threshold, binary):
            img = img0[
                i * patch : min((i + 1) * patch, im_shape[0]),
                j * patch : min((j + 1) * patch, im_shape[1]),
            ]
            if inpaint:
                mask_, clean_ = self.clean_(img, threshold=threshold,
                                            inpaint=True, binary=binary)
                mask[
                    i * patch : min((i + 1) * patch, im_shape[0]),
                    j * patch : min((j + 1) * patch, im_shape[1]),
                ] = mask_
                img1[
                    i * patch : min((i + 1) * patch, im_shape[0]),
                    j * patch : min((j + 1) * patch, im_shape[1]),
                ] = clean_
            else:
                mask_ = self.clean_(img, threshold=threshold,
                                    inpaint=False, binary=binary)
                mask[
                    i * patch : min((i + 1) * patch, im_shape[0]),
                    j * patch : min((j + 1) * patch, im_shape[1]),
                ] = mask_

        _ = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(fill_values)(
                i, j, img0, img1, mask, patch, inpaint, threshold, binary
            )
            for i in range(hh)
            for j in range(ww)
        )

        mask = np.array(mask)
        if inpaint:
            img1 = np.array(img1)
        try:
            shutil.rmtree(folder)
        except Exception:
            print("Could not clean-up automatically.")

        if inpaint:
            return mask, img1
        else:
            return mask

        # serial version
    def clean_large(self, img0, threshold=0.5, inpaint=True, binary=True,
                    patch=256):

        im_shape = img0.shape
        hh = int(math.ceil(im_shape[0] / patch))
        ww = int(math.ceil(im_shape[1] / patch))

        img1 = np.zeros((im_shape[0], im_shape[1]))
        mask = np.zeros((im_shape[0], im_shape[1]))

        if inpaint:
            for i in range(hh):
                for j in range(ww):
                    img = img0[
                        i * patch : min((i + 1) * patch, im_shape[0]),
                        j * patch : min((j + 1) * patch, im_shape[1]),
                    ]
                    mask_, clean_ = self.clean_(img, threshold=threshold,
                                                inpaint=True, binary=binary)
                    mask[
                        i * patch : min((i + 1) * patch, im_shape[0]),
                        j * patch : min((j + 1) * patch, im_shape[1]),
                    ] = mask_
                    img1[
                        i * patch : min((i + 1) * patch, im_shape[0]),
                        j * patch : min((j + 1) * patch, im_shape[1]),
                    ] = clean_
            return mask, img1

        else:
            for i in range(hh):
                for j in range(ww):
                    img = img0[
                        i * patch : min((i + 1) * patch, im_shape[0]),
                        j * patch : min((j + 1) * patch, im_shape[1]),
                    ]
                    mask_ = self.clean_(img, threshold=threshold,
                                        inpaint=False, binary=binary)
                    mask[
                        i * patch : min((i + 1) * patch, im_shape[0]),
                        j * patch : min((j + 1) * patch, im_shape[1]),
                    ] = mask_
            return mask
