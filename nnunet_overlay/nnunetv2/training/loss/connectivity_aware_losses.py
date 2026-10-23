import torch
from torch import nn
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.utilities.helpers import softmax_helper_dim1

class DC_ClassWeightedSkeletonRecallLoss(nn.Module):
    """Dice + CE + class-weighted soft skeleton recall.

    This follows the IMR-style connectivity objective but keeps it differentiable by
    measuring recall with softmax probabilities on ground-truth skeleton voxels.
    """

    def __init__(self, soft_dice_kwargs, weight_ce=1.0, weight_dc=1.0, weight_skel=0.5,
                 small_vessel_labels=(8, 9, 10, 13), small_vessel_weight=1.3,
                 ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss):
        super().__init__()
        self.ignore_label = ignore_label
        self.dc = dice_class(apply_nonlin=softmax_helper_dim1, **soft_dice_kwargs)
        if ignore_label is None:
            self.ce = nn.CrossEntropyLoss()
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.weight_ce = float(weight_ce)
        self.weight_dc = float(weight_dc)
        self.weight_skel = float(weight_skel)
        self.small_vessel_labels = tuple(int(i) for i in small_vessel_labels)
        self.small_vessel_weight = float(small_vessel_weight)

    def _class_weights(self, num_classes: int, device: torch.device, dtype: torch.dtype):
        weights = torch.ones(num_classes, device=device, dtype=dtype)
        for label in self.small_vessel_labels:
            if 0 <= label < num_classes:
                weights[label] = self.small_vessel_weight
        if num_classes > 1:
            fg_mean = weights[1:].mean().clamp_min(1e-6)
            weights[1:] = weights[1:] / fg_mean
        return weights

    def _soft_class_weighted_skeleton_recall(self, net_output: torch.Tensor, skel_label: torch.Tensor):
        probs = softmax_helper_dim1(net_output)
        if skel_label.ndim == probs.ndim:
            skel_label = skel_label[:, 0]
        skel_label = skel_label.long()

        num_classes = probs.shape[1]
        weights = self._class_weights(num_classes, probs.device, probs.dtype)
        recalls = []
        recall_weights = []
        for label in range(1, num_classes):
            mask = skel_label == label
            denom = mask.sum()
            if denom == 0:
                continue
            recall = probs[:, label][mask].sum() / denom.clamp_min(1)
            recalls.append(recall)
            recall_weights.append(weights[label])
        if not recalls:
            # A scale without a reference skeleton must not incur skeleton loss.
            return probs.new_tensor(1.0)
        recalls = torch.stack(recalls)
        recall_weights = torch.stack(recall_weights)
        return (recalls * recall_weights).sum() / recall_weights.sum().clamp_min(1e-6)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor, skel_label: torch.Tensor):
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'ignore label is not implemented for one hot encoded target variables'
            mask = target != self.ignore_label
            target_dice = torch.where(mask, target, 0)
        else:
            target_dice = target
            mask = None

        dc_loss = self.dc(net_output, target_dice, loss_mask=mask) if self.weight_dc != 0 else 0
        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        skel_recall = self._soft_class_weighted_skeleton_recall(net_output, skel_label)
        skel_loss = 1.0 - skel_recall
        return self.weight_dc * dc_loss + self.weight_ce * ce_loss + self.weight_skel * skel_loss

class DeepSupervisionCALWrapper(nn.Module):
    def __init__(self, loss, weight_factors):
        super().__init__()
        self.loss = loss
        self.weight_factors = tuple(weight_factors)

    def forward(self, net_output, target, skel_weight):
        if not isinstance(net_output, (tuple, list)):
            return self.loss(net_output, target, skel_weight)
        weights = self.weight_factors
        result = weights[0] * self.loss(net_output[0], target[0], skel_weight[0])
        for i in range(1, len(net_output)):
            if weights[i] != 0:
                result += weights[i] * self.loss(net_output[i], target[i], skel_weight[i])
        return result
