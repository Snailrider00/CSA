#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize


PCA_LABELS = (2, 3)
PCOM_LABELS = (8, 9)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, n_comp = ndi.label(mask.astype(bool), structure=structure)
    return labeled.astype(np.int32, copy=False), int(n_comp)


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n_comp = connected_components(mask)
    if n_comp <= 1:
        return mask.astype(bool)
    sizes = ndi.sum(mask.astype(np.uint8), labeled, index=np.arange(1, n_comp + 1))
    largest = int(np.argmax(sizes) + 1)
    return labeled == largest


def find_endpoints(skeleton: np.ndarray) -> np.ndarray:
    if not skeleton.any():
        return np.empty((0, 3), dtype=np.int64)
    padded = np.pad(skeleton.astype(bool), 1, mode='constant', constant_values=False)
    endpoints = []
    for idx in np.argwhere(skeleton):
        pi = idx + 1
        block = padded[pi[0] - 1:pi[0] + 2, pi[1] - 1:pi[1] + 2, pi[2] - 1:pi[2] + 2]
        degree = int(np.count_nonzero(block)) - 1
        if degree == 1:
            endpoints.append(idx)
    return np.asarray(endpoints, dtype=np.int64)


def draw_bridge(mask: np.ndarray, point_1: np.ndarray, point_2: np.ndarray, n_points: int = 100) -> np.ndarray:
    points = np.vstack([point_1, point_2]).astype(float)
    mid_point_1 = (point_1 + point_2) / 2.0
    mid_point_2 = (mid_point_1 + point_1) / 2.0
    points = np.vstack([point_1, mid_point_2, mid_point_1, point_2]).astype(float)
    try:
        tck, _ = splprep([points[:, 0], points[:, 1], points[:, 2]], s=0)
        new_points = np.asarray(splev(np.linspace(0, 1, n_points), tck)).T
    except Exception:
        new_points = np.linspace(point_1, point_2, n_points)
    new_points = np.round(new_points).astype(np.int64)
    out = mask.astype(bool).copy()
    shape = np.asarray(out.shape)
    valid = np.all((new_points >= 0) & (new_points < shape), axis=1)
    for x, y, z in new_points[valid]:
        out[x, y, z] = True
    return out


def postprocess_pcom(mask: np.ndarray, component_threshold: int = 20, distance_threshold: float = 20.0) -> tuple[np.ndarray, str]:
    labeled, n_comp = connected_components(mask)
    if n_comp == 0:
        return mask.astype(bool), 'empty'
    if n_comp == 1:
        return mask.astype(bool), 'single_component_keep'
    if n_comp == 2:
        comp1 = labeled == 1
        comp2 = labeled == 2
        size1 = int(np.count_nonzero(comp1))
        size2 = int(np.count_nonzero(comp2))
        if size1 >= component_threshold and size2 >= component_threshold:
            skel1 = skeletonize(comp1)
            skel2 = skeletonize(comp2)
            endpoints1 = find_endpoints(skel1)
            endpoints2 = find_endpoints(skel2)
            if len(endpoints1) and len(endpoints2):
                dist = cdist(endpoints1, endpoints2)
                min_index = np.unravel_index(np.argmin(dist), dist.shape)
                point_1 = endpoints1[min_index[0]]
                point_2 = endpoints2[min_index[1]]
                min_distance = float(np.linalg.norm(point_2 - point_1))
                if min_distance <= distance_threshold:
                    return draw_bridge(mask, point_1, point_2), f'bridge_two_components_distance_{min_distance:.2f}'
                return keep_largest_component(mask), f'two_components_far_keep_largest_distance_{min_distance:.2f}'
            return keep_largest_component(mask), 'two_components_no_endpoints_keep_largest'
        if size1 < component_threshold and size2 < component_threshold:
            return np.zeros_like(mask, dtype=bool), 'two_small_components_zero'
        return keep_largest_component(mask), 'one_small_component_keep_largest'
    return keep_largest_component(mask), f'{n_comp}_components_keep_largest'


def process_case(pred_path: Path, out_path: Path) -> dict[str, Any]:
    img = nib.load(str(pred_path))
    arr = np.asarray(img.dataobj).astype(np.uint8, copy=True)
    out = arr.copy()
    record: dict[str, Any] = {'case': pred_path.name.replace('.nii.gz', '')}

    for label in PCA_LABELS:
        before = arr == label
        after = keep_largest_component(before)
        out[out == label] = 0
        out[after] = label
        record[f'label_{label}_voxels_before'] = int(np.count_nonzero(before))
        record[f'label_{label}_voxels_after'] = int(np.count_nonzero(after))
        record[f'label_{label}_operation'] = 'keep_largest_component'

    for label in PCOM_LABELS:
        before = out == label
        after, op = postprocess_pcom(before)
        out[out == label] = 0
        out[after] = label
        record[f'label_{label}_voxels_before'] = int(np.count_nonzero(before))
        record[f'label_{label}_voxels_after'] = int(np.count_nonzero(after))
        record[f'label_{label}_operation'] = op

    out_img = nib.Nifti1Image(out.astype(np.uint8), img.affine, img.header)
    out_img.set_data_dtype(np.uint8)
    nib.save(out_img, str(out_path))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--case-prefix', default='topcow_mr_')
    args = parser.parse_args()

    in_dir = args.input_dir.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pred_path in sorted(in_dir.glob(f'{args.case_prefix}*.nii.gz')):
        rows.append(process_case(pred_path, out_dir / pred_path.name))
    if not rows:
        raise RuntimeError(f'No predictions found in {in_dir} with prefix {args.case_prefix}')
    with (out_dir / 'postprocess_operations.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(out_dir)
    print(out_dir / 'postprocess_operations.csv')


if __name__ == '__main__':
    main()
