#!/usr/bin/env python3
"""Restore fold-specific ROI probabilities, average on the full grid, then argmax.

Each manifest entry supplies reference_image and a list of folds containing
probabilities (.npz, key 'probabilities'), roi_image and start_xyz.
Probability axes must be C,Z,Y,X as exported by nnU-Net SimpleITKIO.
"""
import argparse,json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from prepare_data import geometry_equal


def combine(reference, folds, expected_folds=5):
    if len(folds)!=expected_folds:raise ValueError(f'Expected {expected_folds} folds')
    shape=tuple(reference.GetSize()[::-1])
    total=np.zeros((14,*shape),np.float32)
    total[0]=float(len(folds))
    for fold in folds:
        roi=sitk.ReadImage(str(fold['roi_image']))
        start=np.asarray(fold['start_xyz'],dtype=int)
        size=np.asarray(roi.GetSize(),dtype=int)
        if np.any(start<0) or np.any(start+size>reference.GetSize()):raise ValueError('ROI outside full image')
        expected=sitk.RegionOfInterest(reference,size.tolist(),start.tolist())
        if not geometry_equal(expected,roi):raise ValueError('ROI physical geometry mismatch')
        with np.load(fold['probabilities'],allow_pickle=False) as z:prob=z['probabilities'].astype(np.float32)
        if prob.shape!=(14,*size[::-1]) or not np.isfinite(prob).all():raise ValueError('Invalid probability shape or values')
        if prob.min() < -1e-4 or prob.max()>1.0001 or not np.allclose(prob.sum(0),1,atol=2e-3):raise ValueError('Expected normalized softmax probabilities')
        sl=tuple(slice(int(a),int(a+b)) for a,b in zip(start[::-1],size[::-1]))
        total[(0,*sl)]-=1
        total[(slice(None),*sl)]+=prob
    return np.argmax(total/len(folds),axis=0).astype(np.uint8)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    for item in json.loads(args.manifest.read_text()):
        name=item['case']
        if Path(name).name!=name:raise ValueError('Case must be a simple filename stem')
        ref=sitk.ReadImage(str(item['reference_image']))
        out=sitk.GetImageFromArray(combine(ref,item['folds']));out.CopyInformation(ref)
        sitk.WriteImage(out,str(args.output_dir/f'{name}.nii.gz'),True)


if __name__=='__main__':main()
