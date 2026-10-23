"""CPU checks with synthetic inputs only; no patient data or trained weights."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from copy import deepcopy
import numpy as np
import torch
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from prepare_data import crop_bounds, normalize
from ensemble_probabilities import combine
from evaluate import cldice, voxel_metrics, boundary_metrics
from postprocess import keep_largest_component, postprocess_pcom
from train import METHODS
from nnunetv2.training.dataloading.data_loader_skel_weight import nnUNetDataLoaderClassSkel, _make_class_skel_like_target
from nnunetv2.training.loss.connectivity_aware_losses import DC_ClassWeightedSkeletonRecallLoss, DeepSupervisionCALWrapper
from nnunetv2.training.nnUNetTrainer.variants.training_length import nnUNetTrainerTopCoWClean2000 as trainers
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerTopCoWAttentionAblation import _TopCoWAttentionTrainerBase
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import LabelManager

torch.set_num_threads(2)


class SyntheticDataset:
    identifiers = ['synthetic']
    def load_case(self, key):
        seg = np.zeros((1, 9, 9, 9), np.int16)
        seg[0, 1:8, 4, 4] = 8
        return np.zeros_like(seg, dtype=np.float32), seg, None, {'class_locations': {8: np.argwhere(seg == 8)}}


class ReleaseTests(unittest.TestCase):
    def test_patient_splits(self):
        splits = json.loads((ROOT / 'configs/splits_final.json').read_text())
        seen = set()
        for split in splits:
            train, val = (set(split[k]) for k in ('train', 'val'))
            tp, vp = ({c.rsplit('_', 1)[1] for c in g} for g in (train, val))
            self.assertEqual((len(train), len(val), len(tp), len(vp)), (200, 50, 100, 25))
            self.assertFalse(tp & vp)
            self.assertFalse(vp & seen)
            seen |= vp
        self.assertEqual(len(seen), 125)

    def test_training_configs_and_loss(self):
        plan = json.loads((ROOT / 'configs/nnUNetPlans.json').read_text())
        dataset = json.loads((ROOT / 'configs/dataset.json').read_text())
        plan['continue_training'] = False
        for method, name in METHODS.items():
            trainer = getattr(trainers, name)(deepcopy(plan), '3d_fullres', 0, dataset, device=torch.device('cpu'))
            self.assertEqual(trainer.num_epochs, 2000)
            self.assertEqual(trainer.oversample_foreground_percent, 0.33)
            self.assertEqual((trainer.num_iterations_per_epoch, trainer.num_val_iterations_per_epoch), (250, 50))
            self.assertIsNone(trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()[-1])
            loss = trainer._build_loss()
            weights = getattr(loss, 'weight_factors')
            np.testing.assert_allclose(weights, [8/15, 4/15, 2/15, 1/15, 0])
            if method.startswith('CAL') or method == 'CSA':
                self.assertEqual(trainer.weight_skel, {'CAL_skel0.25': 0.25, 'CAL_skel1.0': 1.0}.get(method, 0.5))
                self.assertEqual(trainer.small_vessel_weight, {'CAL_uniform': 1.0, 'CAL_weight1.6': 1.6}.get(method, 1.3))

    def test_attention_forward_and_parameter_counts(self):
        plans = PlansManager(json.loads((ROOT / 'configs/nnUNetPlans.json').read_text()))
        config = plans.get_configuration('3d_fullres')
        counts = {}
        for rsa, eca in [(False, False), (True, False), (False, True), (True, True)]:
            net = _TopCoWAttentionTrainerBase._build_attention_network(plans, config, 1, 14, True, rsa, eca)
            counts[rsa, eca] = sum(p.numel() for p in net.parameters())
            outputs = net(torch.randn(1, 1, 32, 32, 32))
            self.assertEqual(len(outputs), 5)
            self.assertEqual(tuple(outputs[0].shape), (1, 14, 32, 32, 32))
            sum(o.square().mean() for o in outputs).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in net.parameters()))
        base = counts[False, False]
        self.assertEqual(counts[True, False] - base, 1610)
        self.assertEqual(counts[False, True] - base, 4)
        self.assertEqual(counts[True, True] - base, 1614)

    def test_skeleton_and_loss_gradients(self):
        target = torch.zeros(2, 1, 9, 9, 9, dtype=torch.int16)
        target[:, :, 1:8, 4, 4] = 8
        target[:, :, 4, 1:8, 7] = 13
        skel = _make_class_skel_like_target(target)
        self.assertEqual(set(torch.unique(skel).tolist()), {0, 8, 13})
        self.assertTrue(torch.all(target[skel > 0] == skel[skel > 0]))
        logits = torch.randn(2, 14, 9, 9, 9, requires_grad=True)
        loss = DC_ClassWeightedSkeletonRecallLoss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False})
        value = loss(logits, target, skel)
        value.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(loss._soft_class_weighted_skeleton_recall(logits, torch.zeros_like(skel))), 1.0)
        wrapped = DeepSupervisionCALWrapper(loss, (1, 0))
        self.assertTrue(torch.allclose(wrapped([logits, None], [target, None], [skel, None]), value))

    def test_loader_generates_skeleton_after_transform(self):
        def transform(image, segmentation):
            return {'image': image, 'segmentation': [segmentation, segmentation[:, ::2, ::2, ::2]]}
        loader = nnUNetDataLoaderClassSkel(SyntheticDataset(), 2, (9, 9, 9), (9, 9, 9),
            LabelManager({'background': 0, 'vessel': 8}, None), transforms=transform)
        batch = loader.generate_train_batch()
        self.assertEqual(len(batch['skel']), 2)
        for target, skel in zip(batch['target'], batch['skel']):
            self.assertEqual(target.shape, skel.shape)
            self.assertTrue(torch.all(target[skel > 0] == skel[skel > 0]))

    def test_metric_empty_and_spacing(self):
        a = np.zeros((9, 9, 9), bool); a[2:7, 4, 4] = True
        z = np.zeros_like(a)
        self.assertEqual(cldice(a, a), 1)
        self.assertEqual(cldice(a, z), 0)
        self.assertTrue(math.isnan(cldice(z, z)))
        self.assertEqual(voxel_metrics(a, z)['dice'], 0)
        b = np.roll(a, 1, axis=1)
        distances = boundary_metrics(a, b, (1, 2, 1))
        self.assertEqual(distances, {'hd95_mm': 2, 'assd_mm': 2})
        self.assertTrue(math.isnan(boundary_metrics(a, z, (1, 1, 1))['hd95_mm']))

    def test_roi_and_ensemble_geometry(self):
        image = sitk.Image([8, 8, 8], sitk.sitkFloat32)
        image.SetSpacing((0.5, 1, 2)); image.SetOrigin((4, 5, 6))
        mask = sitk.Image(image.GetSize(), sitk.sitkUInt8); mask.CopyInformation(image)
        self.assertEqual(crop_bounds(image, mask), ([0, 0, 0], [8, 8, 8]))
        self.assertTrue(np.all(normalize(np.ones((2, 2))) == 0))
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            roi = sitk.RegionOfInterest(image, [2, 2, 2], [2, 3, 4])
            sitk.WriteImage(roi, str(folder/'roi.nii.gz'))
            prob = np.zeros((14, 2, 2, 2), np.float32); prob[8] = 1
            np.savez_compressed(folder/'p.npz', probabilities=prob)
            fold = {'roi_image': folder/'roi.nii.gz', 'probabilities': folder/'p.npz', 'start_xyz': [2, 3, 4]}
            out = combine(image, [fold]*5)
            self.assertEqual(np.count_nonzero(out == 8), 8)
            self.assertTrue(np.all(out[4:6, 3:5, 2:4] == 8))
            with self.assertRaises(ValueError):combine(image, [fold]*4)
            with self.assertRaises(ValueError):combine(image, [dict(fold, start_xyz=[1, 3, 4])]*5)

    def test_postprocessing_components(self):
        mask = np.zeros((12, 12, 12), bool); mask[1:4, 1:4, 1:4] = True; mask[10, 10, 10] = True
        self.assertEqual(keep_largest_component(mask).sum(), 27)
        self.assertEqual(postprocess_pcom(mask)[0].sum(), 27)
        mask = np.zeros_like(mask); mask[1, 1, 1] = mask[10, 10, 10] = True
        self.assertEqual(postprocess_pcom(mask)[0].sum(), 0)


if __name__ == '__main__':
    unittest.main()
