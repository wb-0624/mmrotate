# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import math
from mmdet.core.bbox.samplers.base_sampler import BaseSampler
from mmdet.core.bbox.samplers.sampling_result import SamplingResult

from ..builder import ROTATED_BBOX_SAMPLERS


@ROTATED_BBOX_SAMPLERS.register_module()
class CLSBalancedSampler(BaseSampler):
    """Random sampler.

    Args:
        num (int): Number of samples
        pos_fraction (float): Fraction of positive samples
        neg_pos_up (int, optional): Upper bound number of negative and
            positive samples. Defaults to -1.
        add_gt_as_proposals (bool, optional): Whether to add ground truth
            boxes as proposals. Defaults to True.
    """

    def __init__(self,
                 num,
                 pos_fraction,
                 neg_pos_ub=-1,
                 add_gt_as_proposals=True,
                 floor_thr=-1,
                 floor_fraction=0,
                 num_bins=3,
                 extreme_ratio=[0.1,0.1], # easy, hard
                 bboxes_type='hbb',
                 **kwargs):
        from mmdet.core.bbox import demodata
        super(CLSBalancedSampler, self).__init__(num, pos_fraction, neg_pos_ub,
                                             add_gt_as_proposals)

        assert floor_thr >= 0 or floor_thr == -1
        assert 0 <= floor_fraction <= 1
        assert num_bins >= 1

        self.floor_thr = floor_thr
        self.floor_fraction = floor_fraction
        self.num_bins = num_bins
        self.rng = demodata.ensure_rng(kwargs.get('rng', None))
        self.bboxes_type = bboxes_type
        self.extreme_ratio = extreme_ratio

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            gallery = torch.tensor(
                gallery, dtype=torch.long, device=torch.cuda.current_device())
        perm = torch.randperm(gallery.numel(), device=gallery.device)[:num]
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice_pos_cls_balance(pos_inds, num_expected, assign_result.labels, assign_result.max_overlaps)

    def random_choice_pos_cls_balance(self, gallery, num, labels, max_overlaps):
        """Random select some elements balance cls from the gallery.

        If `gallery` is a Tensor, the returned indices will be a Tensor;
        If `gallery` is a ndarray or list, the returned indices will be a
        ndarray.

        Args:
            gallery (Tensor | ndarray | list): indices pool.
            num (int): expected sample num.

        Returns:
            Tensor or ndarray: sampled indices.
        """
        assert len(gallery) >= num

        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            gallery = torch.tensor(
                gallery, dtype=torch.long, device=torch.cuda.current_device())
        unique_labels = labels.unique()
        unique_labels = unique_labels[unique_labels > -1]
        samples_per_class = num // len(unique_labels)

        rand_inds = torch.tensor([], dtype=torch.long, device=gallery.device)

        for label in unique_labels:
            if label <= 0:
                continue
            # 获取当前类别的样本索引
            label_indices = (labels == label).nonzero(as_tuple=True)[0]
            if len(label_indices) > samples_per_class:
                # 随机采样 samples_per_class 个样本
                easy_samples = math.floor(self.extreme_ratio[0] * samples_per_class)
                hard_samples = math.floor(self.extreme_ratio[1] * samples_per_class)
                other_samples = samples_per_class - hard_samples - easy_samples

                max_overlaps_label = max_overlaps[label_indices]
                # 获取当前类别的样本索引
                label_indices = (labels == label).nonzero(as_tuple=True)[0]
                max_overlaps_label = max_overlaps[label_indices]

                # 获取排序后的索引
                sorted_indices = max_overlaps_label.argsort()

                # 获取难样本和易样本的索引
                hard_indices = sorted_indices[:hard_samples]  # 难样本：IoU 最低
                easy_indices = sorted_indices[-easy_samples:]  # 易样本：IoU 最高

                # 基于排序后的索引在 max_overlaps 中获取对应的值
                hard_indices = label_indices[hard_indices]
                easy_indices = label_indices[easy_indices]
                
                remaining_mask = ~torch.isin(torch.arange(len(label_indices), device=gallery.device), torch.cat([hard_indices, easy_indices]))
                gallery_o = label_indices[remaining_mask]
                other_indices = gallery_o[torch.randperm(len(gallery_o))[:other_samples]]

                rand_inds = torch.cat([rand_inds, other_indices, hard_indices, easy_indices])
            else:
                rand_inds = torch.cat([rand_inds, label_indices])

        # 如果采样总数不够 num_samples，则在剩下的样本中进行随机采样
        if len(rand_inds) < num:
            # 还差多少个样本
            remain_num = num - len(rand_inds)
            # gallery中除去已经采样的样本
            mask = ~torch.isin(torch.arange(len(gallery), device=gallery.device), rand_inds)
            gallery_o = gallery[mask]
            # 从剩下的样本中随机采样 remain_num 个样本
            need_perm = gallery_o[torch.randperm(len(gallery_o))[:remain_num]]
            rand_inds = torch.cat([rand_inds, need_perm])
        
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds 
    
    def sample_via_interval(self, max_overlaps, full_set, num_expected):
        max_iou = max_overlaps.max()
        iou_interval = (max_iou - self.floor_thr) / self.num_bins
        per_num_expected = int(num_expected / self.num_bins)

        sampled_inds = []
        for i in range(self.num_bins):
            start_iou = self.floor_thr + i * iou_interval
            end_iou = self.floor_thr + (i + 1) * iou_interval
            tmp_set = set(
                np.where(
                    np.logical_and(max_overlaps >= start_iou,
                                   max_overlaps < end_iou))[0])
            tmp_inds = list(tmp_set & full_set)
            if len(tmp_inds) > per_num_expected:
                tmp_sampled_set = self.random_choice(tmp_inds,
                                                     per_num_expected)
            else:
                tmp_sampled_set = np.array(tmp_inds, dtype=int)
            sampled_inds.append(tmp_sampled_set)

        sampled_inds = np.concatenate(sampled_inds)
        if len(sampled_inds) < num_expected:
            num_extra = num_expected - len(sampled_inds)
            extra_inds = np.array(list(full_set - set(sampled_inds)))
            if len(extra_inds) > num_extra:
                extra_inds = self.random_choice(extra_inds, num_extra)
            sampled_inds = np.concatenate([sampled_inds, extra_inds])

        return sampled_inds

    def _sample_neg(self, assign_result, num_expected, **kwargs):
        neg_inds = torch.nonzero(assign_result.gt_inds == 0)
        if neg_inds.numel() != 0:
            neg_inds = neg_inds.squeeze(1)
        if len(neg_inds) <= num_expected:
            return neg_inds
        else:
            max_overlaps = assign_result.max_overlaps.cpu().numpy()
            # balance sampling for negative samples
            neg_set = set(neg_inds.cpu().numpy())

            if self.floor_thr > 0:
                floor_set = set(
                    np.where(
                        np.logical_and(max_overlaps >= 0,
                                       max_overlaps < self.floor_thr))[0])
                iou_sampling_set = set(
                    np.where(max_overlaps >= self.floor_thr)[0])
            elif self.floor_thr == 0:
                floor_set = set(np.where(max_overlaps == 0)[0])
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
            else:
                floor_set = set()
                iou_sampling_set = set(
                    np.where(max_overlaps > self.floor_thr)[0])
                self.floor_thr = 0

            floor_neg_inds = list(floor_set & neg_set)
            iou_sampling_neg_inds = list(iou_sampling_set & neg_set)
            num_expected_iou_sampling = int(num_expected *
                                            (1 - self.floor_fraction))
            if len(iou_sampling_neg_inds) > num_expected_iou_sampling:
                if self.num_bins >= 2:
                    iou_sampled_inds = self.sample_via_interval(
                        max_overlaps, set(iou_sampling_neg_inds),
                        num_expected_iou_sampling)
                else:
                    iou_sampled_inds = self.random_choice(
                        iou_sampling_neg_inds, num_expected_iou_sampling)
            else:
                iou_sampled_inds = np.array(
                    iou_sampling_neg_inds, dtype=int)
            num_expected_floor = num_expected - len(iou_sampled_inds)
            if len(floor_neg_inds) > num_expected_floor:
                sampled_floor_inds = self.random_choice(
                    floor_neg_inds, num_expected_floor)
            else:
                sampled_floor_inds = np.array(floor_neg_inds, dtype=int)
            sampled_inds = np.concatenate(
                (sampled_floor_inds, iou_sampled_inds))
            if len(sampled_inds) < num_expected:
                num_extra = num_expected - len(sampled_inds)
                extra_inds = np.array(list(neg_set - set(sampled_inds)))
                if len(extra_inds) > num_extra:
                    extra_inds = self.random_choice(extra_inds, num_extra)
                sampled_inds = np.concatenate((sampled_inds, extra_inds))
            sampled_inds = torch.from_numpy(sampled_inds).long().to(
                assign_result.gt_inds.device)
            return sampled_inds
    def sample(self,
               assign_result,
               bboxes,
               gt_bboxes,
               gt_labels=None,
               **kwargs):
        """Sample positive and negative bboxes.

        This is a simple implementation of bbox sampling given candidates,
        assigning results and ground truth bboxes.

        Args:
            assign_result (:obj:`AssignResult`): Bbox assigning results.
            bboxes (torch.Tensor): Boxes to be sampled from.
            gt_bboxes (torch.Tensor): Ground truth bboxes.
            gt_labels (Tensor, optional): Class labels of ground truth bboxes.

        Returns:
            :obj:`SamplingResult`: Sampling result.

        Example:
            >>> from mmdet.core.bbox import RandomSampler
            >>> from mmdet.core.bbox import AssignResult
            >>> from mmdet.core.bbox.demodata import ensure_rng, random_boxes
            >>> rng = ensure_rng(None)
            >>> assign_result = AssignResult.random(rng=rng)
            >>> bboxes = random_boxes(assign_result.num_preds, rng=rng)
            >>> gt_bboxes = random_boxes(assign_result.num_gts, rng=rng)
            >>> gt_labels = None
            >>> self = RandomSampler(num=32, pos_fraction=0.5, neg_pos_ub=-1,
            >>>                      add_gt_as_proposals=False)
            >>> self = self.sample(assign_result, bboxes, gt_bboxes, gt_labels)
        """
        if len(bboxes.shape) < 2:
            bboxes = bboxes[None, :]

        if self.bboxes_type == 'obb':
            bboxes = bboxes[:, :5]
        elif self.bboxes_type == 'hbb':
            bboxes = bboxes[:, :4]
        else:
            raise ValueError('bboxes_type must be one of ["hbb", "obb"]')

        gt_flags = bboxes.new_zeros((bboxes.shape[0], ), dtype=torch.uint8)
        if self.add_gt_as_proposals and len(gt_bboxes) > 0:
            if gt_labels is None:
                raise ValueError(
                    'gt_labels must be given when add_gt_as_proposals is True')
            bboxes = torch.cat([gt_bboxes, bboxes], dim=0)
            assign_result.add_gt_(gt_labels)
            gt_ones = bboxes.new_ones(gt_bboxes.shape[0], dtype=torch.uint8)
            gt_flags = torch.cat([gt_ones, gt_flags])

        num_expected_pos = int(self.num * self.pos_fraction)
        pos_inds = self.pos_sampler._sample_pos(
            assign_result, num_expected_pos, bboxes=bboxes, **kwargs)
        # We found that sampled indices have duplicated items occasionally.
        # (may be a bug of PyTorch)
        pos_inds = pos_inds.unique()
        num_sampled_pos = pos_inds.numel()
        num_expected_neg = self.num - num_sampled_pos
        if self.neg_pos_ub >= 0:
            _pos = max(1, num_sampled_pos)
            neg_upper_bound = int(self.neg_pos_ub * _pos)
            if num_expected_neg > neg_upper_bound:
                num_expected_neg = neg_upper_bound
        neg_inds = self.neg_sampler._sample_neg(
            assign_result, num_expected_neg, bboxes=bboxes, **kwargs)
        neg_inds = neg_inds.unique()

        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
