import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from skimage.morphology import skeletonize
from threadpoolctl import threadpool_limits
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

def _class_skeleton_label(segmentation: torch.Tensor) -> torch.Tensor:
    """Create a label-preserving skeleton map for one target sample."""
    seg_np = segmentation.detach().cpu().numpy()
    if seg_np.ndim < 4:
        raise RuntimeError(f"Expected segmentation with channel and spatial axes, got {seg_np.shape}")

    label_map = seg_np[0].astype(np.int16)
    skel_label = np.zeros_like(label_map, dtype=np.int16)
    for label in np.unique(label_map):
        if label <= 0:
            continue
        class_mask = label_map == label
        if class_mask.any():
            class_skel = skeletonize(class_mask)
            skel_label[class_skel] = label
    return torch.from_numpy(skel_label[None])

def _class_skeleton_batch(segmentation: torch.Tensor) -> torch.Tensor:
    """Create label-preserving skeleton maps for a batch target tensor."""
    if segmentation.ndim == 4:
        return _class_skeleton_label(segmentation)
    if segmentation.ndim != 5:
        raise RuntimeError(f"Expected target as [C, *spatial] or [B, C, *spatial], got {segmentation.shape}")
    samples = [_class_skeleton_label(segmentation[i]) for i in range(segmentation.shape[0])]
    return torch.stack(samples, dim=0)

def _make_class_skel_like_target(target):
    if isinstance(target, list):
        return [_class_skeleton_batch(t) for t in target]
    return _class_skeleton_batch(target)

class nnUNetDataLoaderClassSkel(nnUNetDataLoader):
    """Standard sampling with class skeletons generated after augmentation."""

    def generate_train_batch(self):
        selected_keys = list(self.get_indices())
        data_all = None
        seg_all = None
        skel_all = None

        with torch.no_grad():
            with threadpool_limits(limits=1, user_api=None):
                for j, i in enumerate(selected_keys):
                    force_fg = self.get_do_oversample(j)

                    data, seg, seg_prev, properties = self._data.load_case(i)
                    shape = data.shape[1:]

                    bbox_lbs, bbox_ubs = self.get_bbox(
                        shape, force_fg, properties['class_locations']
                    )
                    bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

                    data_cropped = torch.from_numpy(crop_and_pad_nd(data, bbox, 0)).float()
                    seg_cropped = torch.from_numpy(
                        crop_and_pad_nd(seg, bbox, -1, cast_cropped_to=np.int16)
                    ).to(torch.int16)
                    if seg_prev is not None:
                        seg_prev_cropped = torch.from_numpy(
                            crop_and_pad_nd(seg_prev, bbox, -1, cast_cropped_to=np.int16)
                        ).to(torch.int16)
                        seg_cropped = torch.cat((seg_cropped, seg_prev_cropped[None]), dim=0)

                    if self.patch_size_was_2d:
                        data_cropped = data_cropped[:, 0]
                        seg_cropped = seg_cropped[:, 0]

                    if self.transforms is not None:
                        transformed = self.transforms(**{'image': data_cropped, 'segmentation': seg_cropped})
                        data_sample = transformed['image']
                        seg_sample = transformed['segmentation']
                    else:
                        data_sample = data_cropped
                        seg_sample = seg_cropped

                    skel_sample = _make_class_skel_like_target(seg_sample)

                    if data_all is None:
                        data_all = torch.empty((self.batch_size, *data_sample.shape), dtype=torch.float32)
                    data_all[j] = data_sample

                    if isinstance(seg_sample, list):
                        if seg_all is None:
                            seg_all = [torch.empty((self.batch_size, *s.shape), dtype=s.dtype) for s in seg_sample]
                            skel_all = [torch.empty((self.batch_size, *s.shape), dtype=torch.float32) for s in skel_sample]
                        for s_idx, s in enumerate(seg_sample):
                            seg_all[s_idx][j] = s
                            skel_all[s_idx][j] = skel_sample[s_idx]
                    else:
                        if seg_all is None:
                            seg_all = torch.empty((self.batch_size, *seg_sample.shape), dtype=seg_sample.dtype)
                            skel_all = torch.empty((self.batch_size, *skel_sample.shape), dtype=torch.float32)
                        seg_all[j] = seg_sample
                        skel_all[j] = skel_sample

        return {
            'data': data_all, 'target': seg_all, 'skel': skel_all, 'keys': selected_keys,
        }
