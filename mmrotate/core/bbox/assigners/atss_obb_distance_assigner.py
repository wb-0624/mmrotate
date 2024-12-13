# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.ops import points_in_polygons
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

from ..builder import ROTATED_BBOX_ASSIGNERS
from ..iou_calculators import build_iou_calculator
from ..transforms import obb2poly

@ROTATED_BBOX_ASSIGNERS.register_module()
class ATSSDObbAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): Number of bbox selected in each level.
    """

    def __init__(self,
                 topk,
                 angle_version='oc',
                 iou_calculator=dict(type='RBboxOverlaps2D'),
                 correct_list=[1/2, 1/2]):
        self.topk = topk
        self.angle_version = angle_version
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.correct_list = correct_list

    def assign(self,
               bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox and gt
        2. compute center distance between all bbox and gt
        3. for each gt, select k bbox whose center are closest to the gt center
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 5).
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 5).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]

        # compute iou between all bbox and gt
        iou = self.iou_calculator(bboxes, gt_bboxes)
        # 合并计算 overlap 和宽高比偏移
        iou_offset = self.ratio_correct_list[0] * iou

        # 计算bboxes的中心点
        center_points =  self.get_bbox_center(bboxes)# 简化计算
        center_points_gt =  self.get_bbox_center(gt_bboxes) # 简化计算
        gt_widths, gt_heights = self.get_bbox_wh(gt_bboxes)
        
        # 计算中心点对于gt 宽高偏移
        x_offset = ((center_points[:, 0].unsqueeze(0) - center_points_gt[:, 0].unsqueeze(1)) ** 2) / gt_widths.unsqueeze(1)
        y_offset = ((center_points[:, 1].unsqueeze(0) - center_points_gt[:, 1].unsqueeze(1)) ** 2) / gt_heights.unsqueeze(1)
        
        # 合并中心点偏移计算
        center_offset = self.ratio_correct_list[1] * torch.exp(-torch.sqrt(x_offset + y_offset))

        # 最终的overlap
        overlaps = torch.stack([iou_offset, center_offset]).sum(dim=0)
        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        # compute center distance between all bbox and gt
        # the center of gt and bbox
        gt_points = gt_bboxes[:, :2]
        bboxes_points = bboxes[:, :2]

        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        _, topk_idxs = distances.topk(self.topk, dim=0, largest=False)
        candidate_idxs = topk_idxs.t()

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        gt_bboxes = obb2poly(gt_bboxes, self.angle_version)

        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]

        # limit the positive sample's center in gt
        inside_flag = points_in_polygons(bboxes_points, gt_bboxes)
        is_in_gts = inside_flag[candidate_idxs,
                                torch.arange(num_gt)].to(is_pos.dtype)

        is_pos = is_pos & is_in_gts
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

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
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def get_bbox_wh(self, bbox):
        if bbox.size(-1) == 4:
            w = bbox[..., 2] - bbox[..., 0]
            h = bbox[..., 3] - bbox[..., 1]
            return w, h
        elif bbox.size(-1) == 5:
            # (cx, cy, w, h, angle)
            w = bbox[..., 2]
            h = bbox[..., 3]
            return w, h
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
            return cx, cy
        elif bbox.size(-1) == 5:
            # (cx, cy, w, h, angle)
            return bbox[..., 0], bbox[..., 1]
        elif bbox.size(-1) == 8:
            # (x1, y1, x2, y2, x3, y3, x4, y4)
            cx = (bbox[..., 0] + bbox[..., 2] + bbox[..., 4] + bbox[..., 6]) / 4
            cy = (bbox[..., 1] + bbox[..., 3] + bbox[..., 5] + bbox[..., 7]) / 4
            return (cx, cy)

    def get_bbox_ratio(self, w, h):
        return torch.min(w,h) / torch.max(w,h)