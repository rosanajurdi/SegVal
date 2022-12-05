#!/usr/bin/env python3.6

from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path
from random import random

import numpy as np
import scipy as sp
import torch
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.distance import directed_hausdorff

from torch import Tensor
from torch import einsum
from tqdm import tqdm
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
from surface_distance.metrics import compute_robust_hausdorff, compute_surface_distances
# functions redefinitions
tqdm_ = partial(tqdm, ncols=175,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)

def flatten_(to_flat: Iterable[Iterable[A]]) -> List[A]:
    return [e for l in to_flat for e in l]
def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def uc_(fn: Callable) -> Callable:
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    return fn(*args)


def id_(x):
    return x



def static_laplacian(width: int, height: int,
                     kernel: Tuple = None,
                     device=None) -> Tensor:
        """
        This function compute the weights of the graph representing img.
        The weights 0 <= w_i <= 1 will be determined from the difference between the nodes: 1 for identical value,
        0 for completely different.
        :param img: The image, as a (n,n) matrix.
        :param kernel: A binary mask of (k,k) shape.
        :param sigma: Parameter for the weird exponential at the end.
        :param eps: Other parameter for the weird exponential at the end.
        :return: A float valued (n^2,n^2) symmetric matrix. Diagonal is empty
        """
        kernel_: np.ndarray
        if kernel is None:
                kernelSize = 3

                kernel_ = np.ones((kernelSize,) * 2)
                kernel_[(kernelSize // 2,) * 2] = 0

        else:
                kernel_ = np.asarray(kernel)
        # print(kernel_)

        img_shape = (width, height)
        N = width * height

        KW, KH = kernel_.shape
        K = int(np.sum(kernel_))  # 0 or 1

        A = np.pad(np.arange(N).reshape(img_shape),
                   ((KW // 2, KW // 2), (KH // 2, KH // 2)),
                   'constant',
                   constant_values=-1)
        neighs = np.zeros((K, N), np.int64)

        k = 0
        for i in range(KW):
                for j in range(KH):
                        if kernel_[i, j] == 0:
                                continue

                        T = A[i:i + width, j:j + height]
                        neighs[k, :] = T.ravel()
                        k += 1

        T1 = np.tile(np.arange(N), K)
        T2 = neighs.flatten()
        Z = T1 <= T2
        T1, T2 = T1[Z], T2[Z]

        diff = np.ones(len(T1))
        M = sp.sparse.csc_matrix((diff, (T1, T2)), shape=(N, N))
        adj = M + M.T
        laplacian = sp.sparse.spdiags(adj.sum(0), 0, N, N) - adj
        coo_laplacian = laplacian.tocoo()

        indices: Tensor = torch.stack([torch.from_numpy(coo_laplacian.row), torch.from_numpy(coo_laplacian.col)])
        torch_laplacian = torch.sparse.FloatTensor(indices.type(torch.int64),
                                                   torch.from_numpy(coo_laplacian.data),
                                                   torch.Size([N, N])).to(device)
        assert torch_laplacian.device == device

        return torch_laplacian

# fns
def soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->bc", a)[..., None]


def batch_soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->c", a)[..., None]


def soft_centroid(a: Tensor) -> Tensor:
    b, c, w, h = a.shape

    ws, hs = map_(lambda e: Tensor(e).to(a.device).type(torch.float32), np.mgrid[0:w, 0:h])
    assert ws.shape == hs.shape == (w, h)

    flotted = a.type(torch.float32)
    tot = einsum("bcwh->bc", a).type(torch.float32) + 1e-10

    cw = einsum("bcwh,wh->bc", flotted, ws) / tot
    ch = einsum("bcwh,wh->bc", flotted, hs) / tot

    res = torch.stack([cw, ch], dim=2)
    assert res.shape == (b, c, 2)

    return res

def soft_length(a: Tensor, kernel: Tuple = None) -> Tensor:
    B, K, *img_shape = a.shape

    laplacian: Tensor = static_laplacian(*img_shape, device=a.device, kernel=kernel)
    assert laplacian.dtype == torch.float64
    N, M = laplacian.shape
    assert N == M

    results: Tensor = torch.ones((B, K, 1), dtype=torch.float32, device=a.device)
    for b in range(B):
        for k in range(K):
            flat_slice: Tensor = a[b, k].flatten()[:, None].type(torch.float64)

            assert flat_slice.shape == (N, 1)
            slice_length: Tensor = flat_slice.t().mm(laplacian.mm(flat_slice))

            assert slice_length.shape == (1, 1)
            results[b, k, :] = slice_length[...]

    return results


def contour(x):
    '''
    Differenciable aproximation of contour extraction

    '''
    min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
    max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
    contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
    return contour


def soft_skeletonize(x, thresh_width=10):
    '''
    Differenciable aproximation of morphological skelitonization operaton
    thresh_width - maximal expected width of vessel
    '''
    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
        max_min_pool_x = torch.nn.functional.max_pool2d(min_pool_x, (3, 3), 1, 1)
        contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x

# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


# # Metrics and shitz
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8) -> float:
    assert label.shape == pred.shape
    #assert one_hot(label)
    # assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(torch.float32)
    sum_sizes: Tensor = (einsum(sum_str, [label]) + einsum(sum_str, [pred])).type(torch.float32)

    dices: Tensor = (2 * inter_size + smooth) / (sum_sizes + smooth)

    return dices

def dice_acc_3D(pred, label, n) -> float:
    smooth = 1e-8
    assert label.shape == pred.shape
    label = torch.tensor(label)
    pred = torch.tensor(pred)
    s=label.shape[-1]

    pred_one = [class2one_hot(p,n+1)[0][1:].reshape((1, n,s,s)) for p in list(pred)]
    label = [class2one_hot(p,n+1)[0][1:].reshape((1, n,s,s)) for p in list(label)]
    a = [l & p for l, p in zip(pred_one, label)]
    inter_size = torch.cat(a).sum(axis=[2, 3]).sum(axis=0).type(torch.float32)
    b = [l + p for l, p in zip(pred_one, label)]
    sum_sizes: Tensor = torch.cat(pred_one).sum(axis=[2, 3]).sum(axis=0).type(torch.float32) + \
                        torch.cat(label).sum(axis=[2, 3]).sum(axis=0).type(torch.float32)

    dices: Tensor = ((2 * inter_size) + smooth)/ (sum_sizes+smooth)
    dices = np.round(dices.numpy(), decimals=4)
    return dices
dice_coef = partial(meta_dice, "bcwh->bc")
dice_batch = partial(meta_dice, "bcwh->c")  # used for 3d dice
dice_coef_3D = partial(meta_dice, "bcwh->c")

def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b

def hausdorff_deepmind(preds: Tensor, target: Tensor, spacing: Tensor = None, n: int = None) -> Tensor:
    assert preds.shape == target.shape
    target = torch.tensor(target) # should be slice -class -shape
    preds = torch.tensor(preds) # should be slice -class -shape

    preds = [class2one_hot(p, n) for p in list(preds)]
    target = [class2one_hot(p, n) for p in list(target)]

    preds = torch.tensor(torch.cat(preds))
    target = torch.tensor(torch.cat(target))

    assert one_hot(preds)
    assert one_hot(target)

    B, K, *img_shape = preds.shape

    if spacing is None:
        D: int = 3
        spacing = torch.ones((B, D), dtype=torch.float32)

    hdd = []
    for p, t in zip(np.split(np.array(preds), n, axis=1), np.split(np.array(target), n, axis=1)):
        p = p.squeeze()
        t = t.squeeze()
        dictt = compute_surface_distances(mask_gt=np.array(p).astype(np.bool),
                                      mask_pred=np.array(t).astype(np.bool), spacing_mm=[1,1,1])

        hdd.append(compute_robust_hausdorff(dictt, 95))
        # print(hdd)

    return np.array(hdd[1:])

def hausdorff_medpy(preds: Tensor, target: Tensor, spacing: Tensor = None) -> Tensor:
    assert preds.shape == target.shape
    target = torch.tensor(target)
    preds = torch.tensor(preds)

    preds = [class2one_hot(p, 3) for p in list(preds)]
    target = [class2one_hot(p, 3) for p in list(target)]

    preds = torch.tensor(torch.cat(preds))
    target = torch.tensor(torch.cat(target))

    assert one_hot(preds)
    assert one_hot(target)

    B, K, *img_shape = preds.shape

    if spacing is None:
        D: int = len(img_shape)
        spacing = torch.ones((B, D), dtype=torch.float32)

    assert spacing.shape == (B, len(img_shape))

    res = torch.zeros((B, K), dtype=torch.float32, device=preds.device)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()
    n_spacing = spacing.cpu().numpy()

    for b in range(B):
        # print(spacing[b])
        # if K == 2:
        #     res[b, :] = hd(n_pred[b, 1], n_target[b, 1], voxelspacing=n_spacing[b])
        #     continue

        for k in range(K):
            if not n_target[b, k].any():  # No object to predict
                if n_pred[b, k].any():  # Predicted something nonetheless
                    res[b, k] = sum((dd * d)**2 for (dd, d) in zip(n_spacing[b], img_shape)) ** 0.5
                    continue
                else:
                    res[b, k] = 0
                    continue
            if not n_pred[b, k].any():
                if n_target[b, k].any():
                    res[b, k] = sum((dd * d)**2 for (dd, d) in zip(n_spacing[b], img_shape)) ** 0.5
                    continue
                else:
                    res[b, k] = 0
                    continue

            res[b, k] = hd(n_pred[b, k], n_target[b, k], voxelspacing=n_spacing[b])

    return res

def haussdorf(preds: Tensor, target: Tensor) -> Tensor:
    assert preds.shape == target.shape
    target = torch.tensor(target)
    preds = torch.tensor(preds)

    preds = [class2one_hot(p, 3) for p in list(preds)]
    target = [class2one_hot(p, 3) for p in list(target)]

    preds = torch.tensor(torch.cat(preds))
    target = torch.tensor(torch.cat(target))

    assert one_hot(preds)
    assert one_hot(target)

    B, C, _, _ = preds.shape

    res = torch.zeros((B, C), dtype=torch.float32, device=preds.device)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()

    for b in range(B):
        if C == 2:
            res[b, :] = numpy_haussdorf(n_pred[b, 0], n_target[b, 0])
            continue

        for c in range(C):
            res[b, c] = numpy_haussdorf(n_pred[b, c], n_target[b, c])

    return res


def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:
    assert len(pred.shape) == 2
    assert pred.shape == target.shape

    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    
    
    
    assert sset(seg, list(range(C)))
    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res


# Misc utils
def save_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int) -> None:
    b, w, h = segs.shape  # Since we have the class numbers, we do not need a C axis

    for seg, name in zip(segs, names):
        save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        imsave(str(save_path), seg.cpu().numpy())


def augment(*arrs: Union[np.ndarray, Image.Image]) -> List[Image.Image]:
    imgs: List[Image.Image] = map_(Image.fromarray, arrs) if isinstance(arrs[0], np.ndarray) else list(arrs)

    if random() > 0.5:
        imgs = map_(ImageOps.flip, imgs)
    if random() > 0.5:
        imgs = map_(ImageOps.mirror, imgs)
    if random() > 0.5:
        angle = random() * 90 - 45
        imgs = map_(lambda e: e.rotate(angle), imgs)
    return imgs


def augment_arr(*arrs_a: np.ndarray) -> List[np.ndarray]:
    arrs = list(arrs_a)  # manoucherie type check

    if random() > 0.5:
        arrs = map_(np.flip, arrs)
    if random() > 0.5:
        arrs = map_(np.fliplr, arrs)

    return arrs


def get_center(shape: Tuple, *arrs: np.ndarray) -> List[np.ndarray]:
    def g_center(arr):
        if arr.shape == shape:
            return arr

        dx = (arr.shape[0] - shape[0]) // 2
        dy = (arr.shape[1] - shape[1]) // 2

        if dx == 0 or dy == 0:
            return arr[:shape[0], :shape[1]]

        res = arr[dx:-dx, dy:-dy][:shape[0], :shape[1]]  # Deal with off-by-one errors
        assert res.shape == shape, (res.shape, shape, dx, dy)

        return res

    return [g_center(arr) for arr in arrs]
