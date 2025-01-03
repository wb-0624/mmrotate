import torch

from ..iou_calculators import build_iou_calculator, cious_calculate
from mmcv.ops import points_in_polygons
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from ..builder import ROTATED_BBOX_ASSIGNERS


@ROTATED_BBOX_ASSIGNERS.register_module()
class ATSSCIoUAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox. Each
    proposals will be assigned with `0` or a positive integer indicating the
    ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        scale (float): IoU threshold for positive bboxes.
        pos_num (float): find the nearest pos_num points to gt center in this
        level.
    """

    def __init__(self,
                topk,
                ):
        self.topk = topk


    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes."""

        overlaps = cious_calculate(gt_bboxes, bboxes)
        
        # 计算bboxes的中心点
        center_points =  self.get_bbox_center(bboxes)# 简化计算

        center_points_gt =  self.get_bbox_center(gt_bboxes) # 简化计算

        # 最终的分配结果
        INF = 100000000
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        
        # compute center distance between all bbox and gt
        # the center of gt and bbox
        distances = (center_points[:, None, :] -
                     center_points_gt[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        _, candidate_idxs = distances.topk(self.topk, dim=0, largest=False)
        candidate_idxs = torch.cat([candidate_idxs], dim=0)
        overlaps = overlaps.T
        candidate_overlap = overlaps[candidate_idxs, torch.arange(num_gts)]
        overlaps_mean_per_gt = candidate_overlap.mean(0)
        overlaps_std_per_gt = candidate_overlap.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlap >= overlaps_thr_per_gt[None, :]

        for gt_idx in range(num_gts):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gts, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def get_bbox_wh(self, bbox):
        if bbox.size(-1) == 4:
            w = bbox[..., 2] - bbox[..., 0]
            h = bbox[..., 3] - bbox[..., 1]
        elif bbox.size(-1) == 5:
            # (cx, cy, w, h, angle)
            w = bbox[..., 2]
            h = bbox[..., 3]
        elif bbox.size(-1) == 8:
            # (x1, y1, x2, y2, x3, y3, x4, y4)
            w = torch.sqrt((bbox[..., 0] - bbox[..., 2])**2 +
                           (bbox[..., 0] - bbox[..., 4])**2)
            h = torch.sqrt((bbox[..., 0] - bbox[..., 6])**2 +
                            (bbox[..., 0] - bbox[..., 7])**2)
        return w, h

    def get_bbox_center(self, bbox):
        if bbox.size(-1) == 4:
            cx = (bbox[..., 0] + bbox[..., 2]) / 2
            cy = (bbox[..., 1] + bbox[..., 3]) / 2
        elif bbox.size(-1) == 5:
            # (cx, cy, w, h, angle)
            cx = bbox[..., 0]
            cy = bbox[..., 1]
        elif bbox.size(-1) == 8:
            # (x1, y1, x2, y2, x3, y3, x4, y4)
            cx = (bbox[..., 0] + bbox[..., 2] + bbox[..., 4] + bbox[..., 6]) / 4
            cy = (bbox[..., 1] + bbox[..., 3] + bbox[..., 5] + bbox[..., 7]) / 4
        else:
            # need implement size == 6
            raise TypeError(f'bbox shape should be 4 or 5 or 8, now is{bbox.shape}')
        center = torch.stack((cx, cy), dim=-1)
        return center

    def get_bbox_ratio(self, w, h):
        return torch.min(w,h) / torch.max(w,h)