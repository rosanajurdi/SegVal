#!/usr/bin/env python3.6

import random
import pickle
import argparse
import warnings
from pathlib import Path
from pprint import pprint
from functools import partial
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.io import imsave
from skimage.transform import resize

from utils import mmap_, uc_, map_, augment_arr


def norm_arr(img: np.ndarray) -> np.ndarray:
    casted = img.astype(np.float32)
    shifted = casted - casted.min()
    norm = shifted / shifted.max()
    res = norm

    return res


def get_p_id(path: Path) -> str:
    # res = list(path.parents)[-5].name
    res = path.name.split('.nii')[0]

    return res


import csv


def save_slices(IM_path, gt_path,
                dest_dir: Path, shape: Tuple[int], n_augment: int, discard_negatives: bool,
                flair_dir: str = "flair", t1_dir="t1",
                gt_dir: str = "gt", in_npy_dir="in_npy", gt_npy_dir='gt_npy') -> Tuple[int, int]:
    p_id: str = get_p_id(IM_path)  # gets the patient id
    assert len(set(map_(get_p_id, [IM_path, gt_path]))) == 1
    print(p_id)

    space_dict: Dict[str, Tuple[float, float]] = {}

    # Load the data
    dx, dy, dz = nib.load(str(IM_path)).header.get_zooms()
    # assert dx == dy, (dx, dy)
    flair = np.asarray(nib.load(str(IM_path)).dataobj)
    w, h, _ = flair.shape
    gt = np.asarray(nib.load(str(gt_path)).dataobj).astype(np.float32)
    #corresponds to the number of classes
    assert set(np.unique(gt)) <= set([0., 1., 2.])

    pos: int = (gt == 1 | (gt == 2)).sum()
    neg: int = ((gt == 0)).sum()

    flair = flair.astype(np.float32)
    assert len(set(map_(np.shape, [flair, gt]))) == 1
    assert flair.dtype in [np.float32], flair.dtype
    assert gt.dtype in [np.float32], gt.dtype

    # Normalize and check data content
    norm_flair = norm_arr(flair)  # We need to normalize the whole 3d img, not 2d slices
    norm_gt = gt.astype(np.uint8)
    assert 0 == norm_flair.min() and norm_flair.max() == 1, (norm_flair.min(), norm_flair.max())
    assert np.array_equal(np.unique(gt), np.unique(norm_gt))

    save_dir_flair: Path = Path(dest_dir, flair_dir)
    save_dir_gt: Path = Path(dest_dir, gt_dir)
    save_dir_in_npy: Path = Path(dest_dir, in_npy_dir)
    save_dir_gt_npy: Path = Path(dest_dir, gt_npy_dir)
    save_dirs = [save_dir_flair, save_dir_gt]

    for j in range(flair.shape[-1]):
        flair_s = resize(norm_flair[:, :, j], (256, 256), mode="constant", preserve_range=True,
                         anti_aliasing=False).astype(np.float32)
        # t1_s = resize(norm_t1[:, :, j], (256, 256), mode="constant", preserve_range=True, anti_aliasing=False).astype(np.float32)
        gt_s = resize(norm_gt[:, :, j], (256, 256), mode="constant", preserve_range=True, anti_aliasing=False).astype(
            np.uint8)
        # print(np.unique(gt_s))
        slices = [flair_s, gt_s]
        assert flair_s.shape == gt_s.shape
        # gt_s[np.where(gt_s == 2)] = 0
        # assert set(np.unique(gt_s)).issubset([0, 1]), np.unique(gt_s)

        assert set(np.unique(gt_s)).issubset([0, 1, 2]), np.unique(gt_s)
        if discard_negatives and (gt_s.sum() == 0):
            continue

        for k in range(n_augment + 1):
            if k == 0:
                to_save = slices
            else:
                to_save = map_(np.asarray, augment_arr(*slices))
                assert to_save[0].shape == slices[0].shape, (to_save[0].shape, slices[0].shape)

            filename = f"{p_id}_{k}_{j}"
            space_dict[filename] = (dx * 256 / w, dy * 256 / h)
            for save_dir, data in zip(save_dirs, to_save):
                save_dir.mkdir(parents=True, exist_ok=True)

                if "gt" not in str(save_dir):
                    img = (data * 255).astype(np.uint8)
                else:
                    img = data.astype(np.uint8)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    # imsave(str(Path(save_dir, filename).with_suffix(".png")), img)

            multimodal = np.stack(to_save[:-1])  # Do not include the ground truth
            assert 0 <= multimodal.min() and multimodal.max() <= 1
            save_dir_in_npy.mkdir(parents=True, exist_ok=True)
            save_dir_gt_npy.mkdir(parents=True, exist_ok=True)
            np.save(Path(save_dir_in_npy, filename).with_suffix(".npy"), multimodal)
            np.save(Path(save_dir_gt_npy, filename).with_suffix(".npy"), to_save[-1])
            print(np.unique(to_save[-1]))

    return neg, pos, space_dict


import os


def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)

    # Assume the cleaning up is done before calling the script
    assert src_path.exists()
    assert not dest_path.exists()

    # Get all the file names, avoid the temporal ones
    all_paths: List[Path] = list(src_path.rglob('*.nii.gz'))
    nii_paths: List[Path] = [p for p in all_paths if "_4D" not in str(p)]
    assert len(nii_paths) % 2 == 0, "Number of .nii not multiple of 6, some pairs are broken"

    # For training
    IMG_nii_paths: List[Path] = sorted(p for p in nii_paths if "imagesTr" in str(p))
    gt_nii_paths: List[Path] = sorted(p for p in nii_paths if "labelsTr" in str(p))
    assert len(IMG_nii_paths) == len(gt_nii_paths)
    paths: List[Tuple[Path, ...]] = list(zip(IMG_nii_paths, gt_nii_paths))
    """
    # For validation
    IMG_nii_paths_val: List[Path] = sorted(p for p in nii_paths_val if "imagesTr" in str(p))
    gt_nii_paths_val: List[Path] = sorted(p for p in nii_paths_val if "labelsTr" in str(p))
    assert len(IMG_nii_paths_val) == len(gt_nii_paths_val)
    paths_val: List[Tuple[Path, ...]] = list(zip(IMG_nii_paths_val, gt_nii_paths_val))
    """
    print(f"Found training {len(IMG_nii_paths)} pairs in total")
    pprint(paths[:2])

    validation_paths:  List[Tuple[Path, ...]] = random.sample(paths, args.retain)
    training_paths: List[Tuple[Path, ...]] = [p for p in paths if p not in validation_paths]
    assert set(validation_paths).isdisjoint(set(training_paths))
    assert len(paths) == (len(validation_paths) + len(training_paths))

    for mode, _paths, n_augment in zip(["train", "val"], [training_paths, validation_paths], [args.n_augment, 0]):

        three_paths = list(zip(*_paths))

        dest_dir = Path(dest_path, mode)
        print(f"Slicing {len(three_paths[0])} pairs to {dest_dir}")
        assert len(set(map_(len, three_paths))) == 1

        pfun = partial(save_slices, dest_dir=dest_dir, shape=args.shape, n_augment=n_augment,
                       discard_negatives=args.discard_negatives)
        sizes = mmap_(uc_(pfun), zip(*three_paths))
        all_neg, all_pos = zip(*sizes)
        neg, pos = sum(all_neg), sum(all_pos)
        ratio = pos / neg
        print(f"Ratio between pos/neg: {ratio} ({pos}/{neg})")

        # final_dict = {k: v for space_dict in space_dicts for k, v in space_dict.items()}
        """
        with open(Path(dest_dir, "spacing.pkl"), 'wb') as f:
            pickle.dump(final_dict, f, pickle.HIGHEST_PROTOCOL)
            print(f"Saved spacing dictionnary to {f}")
        """
        # for case_paths in tqdm(list(zip(*three_paths)), ncols=50):
        #     uc_(pfun)(case_paths)



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Slicing parameters')

    parser.add_argument('--source_dir', type=str,
                        default='/Users/rosana.eljurdi/Documents/Confidence_Intervals_Olivier/Task04_Hippocampus/Splits/train')
    parser.add_argument('--dest_dir', type=str,
                        default='/Users/rosana.eljurdi/Documents/Confidence_Intervals_Olivier/Task04_Hippocampus/Splits/fold_1')
    parser.add_argument('--img_dir', type=str, default="imagesTr")
    parser.add_argument('--gt_dir', type=str, default="labelsTr")
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retain', type=int, default=50, help="Number of retained patient for the validation data")
    parser.add_argument('--n_augment', type=int, default=0,
                        help="Number of augmentation to create per image, only for the training set")
    parser.add_argument('--discard_negatives', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)



