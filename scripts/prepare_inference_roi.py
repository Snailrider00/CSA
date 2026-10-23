#!/usr/bin/env python3
"""Crop evaluation images using predicted localizer masks; no reference labels needed."""
import argparse
import json
from pathlib import Path
import SimpleITK as sitk
from prepare_data import crop_bounds, normalize, write_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True, help='Input <case>_0000.nii.gz images')
    parser.add_argument('--masks', type=Path, required=True, help='Predicted <case>.nii.gz localizer masks')
    parser.add_argument('--output', type=Path, required=True, help='New fold-specific ROI directory')
    args = parser.parse_args()
    images = sorted(args.images.glob('*_0000.nii.gz'))
    if not images:
        raise ValueError('No input images found')
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for path in images:
        case = path.name.removesuffix('_0000.nii.gz')
        image = sitk.ReadImage(str(path))
        mask = sitk.ReadImage(str(args.masks / f'{case}.nii.gz'))
        start, size = crop_bounds(image, mask)
        roi = sitk.RegionOfInterest(image, size, start)
        out = args.output / path.name
        write_array(normalize(sitk.GetArrayFromImage(roi)), roi, out)
        records.append({'case': case, 'reference_image': str(path.resolve()),
                        'roi_image': str(out.resolve()), 'start_xyz': start})
    (args.output / 'crop_records.json').write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
