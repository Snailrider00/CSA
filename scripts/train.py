#!/usr/bin/env python3
"""Run one fold with the uniform 2000-epoch training budget."""
import argparse
import os
import random

METHODS = {
    'Baseline': 'nnUNetTrainerTopCoWCleanBaseline2000',
    'CAL': 'nnUNetTrainerTopCoWCleanCAL2000',
    'RSA': 'nnUNetTrainerTopCoWCleanRSA2000',
    'ECA': 'nnUNetTrainerTopCoWCleanECA2000',
    'RSA_ECA': 'nnUNetTrainerTopCoWCleanRSAECA2000',
    'CAL_RSA': 'nnUNetTrainerTopCoWCleanCALRSA2000',
    'CAL_ECA': 'nnUNetTrainerTopCoWCleanCALECA2000',
    'CAL_uniform': 'nnUNetTrainerTopCoWCleanCALUniform2000',
    'CAL_weight1.6': 'nnUNetTrainerTopCoWCleanCALWeight162000',
    'CAL_skel0.25': 'nnUNetTrainerTopCoWCleanCALSkel0252000',
    'CAL_skel1.0': 'nnUNetTrainerTopCoWCleanCALSkel102000',
    'CSA': 'nnUNetTrainerTopCoWCleanFull2000',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True)
    p.add_argument('--fold', required=True, type=int, choices=range(5))
    p.add_argument('--method', choices=METHODS, default='CSA')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    os.environ.setdefault('nnUNet_compile', 'false')
    import numpy as np
    import torch
    from nnunetv2.run.run_training import run_training
    seed = 20260922 + args.fold
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    run_training(args.dataset, '3d_fullres', args.fold,
                 trainer_class_name=METHODS[args.method],
                 continue_training=args.resume, val_with_best=True)


if __name__ == '__main__':
    main()
