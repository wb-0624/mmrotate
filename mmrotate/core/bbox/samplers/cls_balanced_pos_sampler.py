import torch

from mmdet.core.bbox.samplers import RandomSampler


class CLSBalancedPosSampler(RandomSampler):

    def _sample_pos(self, assign_result, num_expected, **kwargs):
        """Randomly sample some positive samples."""
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
        if pos_inds.numel() != 0:
            pos_inds = pos_inds.squeeze(1)
        if pos_inds.numel() <= num_expected:
            return pos_inds
        else:
            return self.random_choice_pos_cls_balance(pos_inds, num_expected, assign_result.labels)

    def random_choice_pos_cls_balance(self, gallery, num, labels):
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
        samples_per_class = num // len(unique_labels)
        perm = []

        for label in unique_labels:
            if label <= 0:
                continue
            # 获取当前类别的样本索引
            label_indices = (labels == label).nonzero(as_tuple=True)[0]
            if len(label_indices) > samples_per_class:
                # 随机采样 samples_per_class 个样本
                perm.extend(label_indices[torch.randperm(len(label_indices))[:samples_per_class]].tolist())
            else:
                perm.extend(label_indices.tolist())

        # 如果采样总数不够 num_samples，则在剩下的样本中进行随机采样
        if len(perm) < num:
            remaining_indices = list(set(range(len(labels))) - set(perm))
            additional_samples = torch.randperm(len(remaining_indices))[:num - len(perm)].tolist()
            perm.extend([remaining_indices[i] for i in additional_samples])

        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds 