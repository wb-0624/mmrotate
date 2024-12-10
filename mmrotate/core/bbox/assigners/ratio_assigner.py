import torch

from ..iou_calculators import build_iou_calculator
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from ..builder import ROTATED_BBOX_ASSIGNERS


@ROTATED_BBOX_ASSIGNERS.register_module()
class RatioAssigner(BaseAssigner):
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
                 pos_iou_thr,
                 neg_iou_thr,
                 min_pos_iou=.0,
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 match_low_quality=True,
                 gpu_assign_thr=-1,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 ratio_correct=[1/3,1/3,1/3]
                 ):
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ratio_correct_list = ratio_correct


    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes."""
        
        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        
        # 在需要时才切换到 CPU
        if assign_on_cpu:
            device = bboxes.device
            bboxes = bboxes.cpu()
            gt_bboxes = gt_bboxes.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()
            if gt_labels is not None:
                gt_labels = gt_labels.cpu()

        iou = self.iou_calculator(gt_bboxes, bboxes)
        
        # 合并计算 overlap 和宽高比偏移
        iou_offset = self.ratio_correct_list[0] * iou

        # 计算bboxes的中心点
        center_points =  self.get_bbox_center(bboxes)# 简化计算
        # 计算宽高比
        widths, heights = self.get_bbox_wh(bboxes)

        center_points_gt =  self.get_bbox_center(gt_bboxes) # 简化计算
        gt_widths, gt_heights = self.get_bbox_wh(gt_bboxes)
        
        # print("=========================")
        # print(center_points)
        # 计算中心点偏移
        x_offset = ((center_points[:, 0].unsqueeze(0) - center_points_gt[:, 0].unsqueeze(1)) ** 2) / gt_widths.unsqueeze(1)
        y_offset = ((center_points[:, 1].unsqueeze(0) - center_points_gt[:, 1].unsqueeze(1)) ** 2) / gt_heights.unsqueeze(1)
        
        # 合并中心点偏移计算
        center_offset = self.ratio_correct_list[1] * torch.exp(-torch.sqrt(x_offset + y_offset))

        # 计算宽高比差距
        bbox_ratio = (torch.min(widths, heights) / torch.max(widths, heights)).unsqueeze(0)
        gt_ratio = (torch.min(gt_widths, gt_heights) / torch.max(gt_widths, gt_heights)).unsqueeze(1)
        ratio = bbox_ratio / gt_ratio
        ratio_offset = self.ratio_correct_list[2] * torch.exp(-torch.abs(torch.log(ratio)))

        # 最终的overlap
        overlaps = torch.stack([iou_offset, center_offset, ratio_offset]).sum(dim=0)
        # print("===================ratio assigner=============================")
        # # print(iou_offset.shape, center_offset.shape, ratio_offset.shape)
        # print("bboxes shape: ",bboxes.shape)
        # print("gt_bboxes: ", gt_bboxes.shape)
        # print("gt_bboxes: ", gt_bboxes)
        # print("overlaps shape: ", overlaps.shape)

        # 处理忽略区域
        if self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None and gt_bboxes_ignore.numel() > 0:
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(bboxes, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(gt_bboxes_ignore, bboxes, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        # 最终的分配结果
        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        
        # 如果之前将张量移到了CPU，需要恢复到原来的设备
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)

        return assign_result



    def assign_wrt_overlaps(self, overlaps, gt_labels=None):
        """Assign w.r.t. the overlaps of bboxes with gts.

        Args:
            overlaps (Tensor): Overlaps between k gt_bboxes and n bboxes,
                shape(k, n).
            gt_labels (Tensor, optional): Labels of k gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             -1,
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

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        # 2. assign negative: below
        # the negative inds are set to be 0
        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < self.neg_iou_thr)] = 0
        elif isinstance(self.neg_iou_thr, tuple):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[(max_overlaps >= self.neg_iou_thr[0])
                             & (max_overlaps < self.neg_iou_thr[1])] = 0

        # 3. assign positive: above positive IoU threshold
        pos_inds = max_overlaps >= self.pos_iou_thr
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        if self.match_low_quality:
            # Low-quality matching will overwirte the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox B.
            # This might be the reason that it is not used in ROI Heads.
            for i in range(num_gts):
                if gt_max_overlaps[i] >= self.min_pos_iou:
                    if self.gt_max_assign_all:
                        max_iou_inds = overlaps[i, :] == gt_max_overlaps[i]
                        assigned_gt_inds[max_iou_inds] = i + 1
                    else:
                        assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

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