#!/usr/bin/env python3
"""Prepare binary localization data or fold-specific, predicted-ROI fine data.

Portable entry point extracted from the local five-fold preparation workflow.
Input: unpacked TopCoW release, imagesTr/ and cow_seg_labelsTr/.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import SimpleITK as sitk

CONFIG = Path(__file__).resolve().parents[1] / 'configs'


def geometry_equal(a, b):
    return a.GetSize() == b.GetSize() and all(np.allclose(getattr(a, k)(), getattr(b, k)(), atol=1e-4)
        for k in ('GetOrigin', 'GetSpacing', 'GetDirection'))


def normalize(array):
    a = np.clip(array.astype(np.float32), -1000, 1800)
    low, high = float(a.min()), float(a.max())
    return (a - low) / (high - low) if high > low else np.zeros_like(a)


def crop_bounds(image, mask, margin=16):
    if not geometry_equal(image, mask):
        raise ValueError('Localizer mask and image physical geometry differ')
    pos = np.where(sitk.GetArrayFromImage(mask) > 0)
    if not len(pos[0]):
        return [0, 0, 0], list(image.GetSize())
    low = np.array([int(x.min()) for x in pos])[::-1]
    high = np.array([int(x.max()) + 1 for x in pos])[::-1]
    start = np.maximum(0, low - margin)
    end = np.minimum(image.GetSize(), high + margin)
    return start.tolist(), (end - start).tolist()


def write_array(a, reference, path):
    out = sitk.GetImageFromArray(a)
    out.CopyInformation(reference)
    sitk.WriteImage(out, str(path), True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path, help='New DatasetNNN_Name directory under nnUNet_raw')
    p.add_argument('--mode', choices=['localizer', 'roi'], required=True)
    p.add_argument('--modality', choices=['ct', 'mr'], help='Required for localizer mode')
    p.add_argument('--masks', type=Path, help='Fold-matched predicted masks, <case>.nii.gz')
    p.add_argument('--splits', type=Path, default=CONFIG/'splits_final.json')
    p.add_argument('--plans', type=Path, default=CONFIG/'nnUNetPlans.json')
    p.add_argument('--preprocessed-root', type=Path, help='Required: nnUNet_preprocessed directory')
    args = p.parse_args()
    if not args.preprocessed_root:
        p.error('--preprocessed-root is required')
    if args.mode == 'localizer' and not args.modality:
        p.error('--modality is required for localizers')
    if args.mode == 'roi' and not args.masks:
        p.error('--masks must contain predictions from the localizer of this fold')
    if args.output.exists():
        raise FileExistsError(f'Output already exists: {args.output}; use a new directory')
    prefix = f'topcow_{args.modality}_' if args.mode == 'localizer' else 'topcow_'
    files = sorted((args.source/'imagesTr').glob(prefix+'*_0000.nii.gz'))
    expected = 125 if args.mode == 'localizer' else 250
    if len(files) != expected:
        raise ValueError(f'Expected {expected} official development volumes, got {len(files)}')
    splits = json.loads(args.splits.read_text())
    cases = {x.name.removesuffix('_0000.nii.gz') for x in files}
    if args.mode == 'localizer':
        splits = [{k:[c for c in s[k] if c.startswith(prefix)] for k in ('train','val')} for s in splits]
    seen = set()
    for s in splits:
        train, val = set(s['train']), set(s['val'])
        tp, vp = ({x.rsplit('_', 1)[1] for x in group} for group in (train,val))
        if train | val != cases or tp & vp or len(tp)!=100 or len(vp)!=25 or seen & vp:
            raise ValueError('Invalid patient-level split')
        seen |= vp
    for name in ['imagesTr','labelsTr']:(args.output/name).mkdir(parents=True)
    records = []
    for path in files:
        case = path.name.removesuffix('_0000.nii.gz')
        im = sitk.ReadImage(str(path))
        label = sitk.ReadImage(str(args.source/'cow_seg_labelsTr'/f'{case}.nii.gz'))
        if not geometry_equal(im,label):raise ValueError(f'Image/label geometry mismatch: {case}')
        if args.mode == 'localizer':
            shutil.copy2(path,args.output/'imagesTr'/path.name)
            write_array((sitk.GetArrayFromImage(label)>0).astype(np.uint8),label,args.output/'labelsTr'/f'{case}.nii.gz')
        else:
            mask = sitk.ReadImage(str(args.masks/f'{case}.nii.gz'))
            start,size = crop_bounds(im,mask)
            roi = sitk.RegionOfInterest(im,size,start)
            lab = sitk.RegionOfInterest(label,size,start)
            a = sitk.GetArrayFromImage(lab)
            if not set(np.unique(a)).issubset(set(range(14))|{15}):raise ValueError(f'Unknown labels: {case}')
            a = a.astype(np.uint8);a[a==15]=13
            write_array(normalize(sitk.GetArrayFromImage(roi)),roi,args.output/'imagesTr'/path.name)
            write_array(a,lab,args.output/'labelsTr'/f'{case}.nii.gz')
            records.append({'case':case,'start_xyz':start,'size_xyz':size})
    dataset = json.loads((CONFIG/'dataset.json').read_text())
    dataset['name'] = args.output.name
    dataset['numTraining'] = len(files)
    if args.mode == 'localizer':
        dataset.update(channel_names={'0':'CT' if args.modality=='ct' else 'MRA'},labels={'background':0,'vessel':1})
        dataset.pop('description',None)
    prep = args.preprocessed_root/args.output.name;prep.mkdir(parents=True,exist_ok=True)
    for directory in [args.output,prep]:
        (directory/'dataset.json').write_text(json.dumps(dataset,indent=2))
        (directory/'splits_final.json').write_text(json.dumps(splits,indent=2))
    if args.mode == 'roi':
        plans = json.loads(args.plans.read_text());plans['dataset_name']=args.output.name
        (prep/'nnUNetPlans.json').write_text(json.dumps(plans,indent=2))
        (args.output/'crop_records.json').write_text(json.dumps(records,indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
