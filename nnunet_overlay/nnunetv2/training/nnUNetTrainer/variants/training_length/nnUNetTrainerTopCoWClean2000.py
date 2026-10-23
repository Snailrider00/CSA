"""Uniform budgets for patient-level five-fold CSA and module ablations."""
import shutil
import time

import torch

from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerNoMirroring import nnUNetTrainerNoMirroring
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerConnectivityAwareTopCoW import nnUNetTrainerIMRClassWeightedSkeletonRecallNoMirroring1000
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerTopCoWAttentionAblation import (
    nnUNetTrainerTopCoWSkipAttentionNoMirroring1000,
    nnUNetTrainerTopCoWECANoMirroring1000,
    nnUNetTrainerTopCoWJointAttentionNoMirroring1000,
    nnUNetTrainerTopCoWCALSkipCleanNoMirroring1000,
    nnUNetTrainerTopCoWCALECACleanNoMirroring1000,
    nnUNetTrainerTopCoWCALJointAttentionCleanNoMirroring1000,
)


class _Clean2000Budget:
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 2000
        self.num_iterations_per_epoch = 250
        self.num_val_iterations_per_epoch = 50
        self.oversample_foreground_percent = 0.33
        self.save_every = 25

    def on_train_epoch_start(self):
        # Check before training another epoch, while space remains to save state.
        if shutil.disk_usage(self.output_folder).free < 3 * 2**30:
            self.save_checkpoint(self.output_folder + "/checkpoint_latest.pth")
            while shutil.disk_usage(self.output_folder).free < 6 * 2**30:
                self.print_to_log_file("Paused for disk space: need 6 GiB free to resume.")
                time.sleep(300)
        return super().on_train_epoch_start()


class nnUNetTrainerTopCoWCleanBaseline2000(_Clean2000Budget, nnUNetTrainerNoMirroring):
    pass


class nnUNetTrainerTopCoWCleanCAL2000(_Clean2000Budget, nnUNetTrainerIMRClassWeightedSkeletonRecallNoMirroring1000):
    pass


class nnUNetTrainerTopCoWCleanRSA2000(_Clean2000Budget, nnUNetTrainerTopCoWSkipAttentionNoMirroring1000):
    pass


class nnUNetTrainerTopCoWCleanECA2000(_Clean2000Budget, nnUNetTrainerTopCoWECANoMirroring1000):
    pass


class nnUNetTrainerTopCoWCleanRSAECA2000(_Clean2000Budget, nnUNetTrainerTopCoWJointAttentionNoMirroring1000):
    pass


class nnUNetTrainerTopCoWCleanCALRSA2000(_Clean2000Budget, nnUNetTrainerTopCoWCALSkipCleanNoMirroring1000):
    pass


class nnUNetTrainerTopCoWCleanCALECA2000(_Clean2000Budget, nnUNetTrainerTopCoWCALECACleanNoMirroring1000):
    pass


class nnUNetTrainerTopCoWCleanFull2000(_Clean2000Budget, nnUNetTrainerTopCoWCALJointAttentionCleanNoMirroring1000):
    pass


class nnUNetTrainerTopCoWCleanCALUniform2000(nnUNetTrainerTopCoWCleanCAL2000):
    """Manuscript uniform-weight CAL comparison."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.small_vessel_weight = 1.0


class nnUNetTrainerTopCoWCleanCALWeight162000(nnUNetTrainerTopCoWCleanCAL2000):
    """Manuscript CAL parameter comparison: target-class weight 1.6."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.small_vessel_weight = 1.6


class nnUNetTrainerTopCoWCleanCALSkel0252000(nnUNetTrainerTopCoWCleanCAL2000):
    """Manuscript CAL parameter comparison: skeleton coefficient 0.25."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight_skel = 0.25


class nnUNetTrainerTopCoWCleanCALSkel102000(nnUNetTrainerTopCoWCleanCAL2000):
    """Manuscript CAL parameter comparison: skeleton coefficient 1.0."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.weight_skel = 1.0
