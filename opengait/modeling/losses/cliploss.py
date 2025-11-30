import math
import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from timm.loss import LabelSmoothingCrossEntropy
from .base import BaseLoss, gather_and_scale_wrapper


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(BaseLoss):

    def __init__(
        self,
        loss_term_weight=1.0,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        smoothing=0.,
    ):
        super().__init__(loss_term_weight=loss_term_weight)
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.label_smoothing_cross_entropy = LabelSmoothingCrossEntropy(smoothing=smoothing) if smoothing > 0 else None

        # cache state
        self.prev_num_logits = 0
        self.labels = {}
    
    @gather_and_scale_wrapper
    def forward(self, image_features, text_features, logit_scale=1.0):
        device = image_features.device

        image_features = image_features.mean(dim=-1)  # [n, c]
        text_features = text_features.mean(dim=-1)

        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad,
                self.rank, self.world_size, self.use_horovod
            )

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T

        # Get or cache labels
        num_logits = logits_per_image.shape[0]
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]

        # Compute the loss
        if self.label_smoothing_cross_entropy:
            loss = (
                self.label_smoothing_cross_entropy(logits_per_image, labels) +
                self.label_smoothing_cross_entropy(logits_per_text, labels)
            ) / 2
        else:
            loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2

        # Compute accuracy and log in self.info
        i2t_acc = (logits_per_image.argmax(-1) == labels).float().mean()
        t2i_acc = (logits_per_text.argmax(-1) == labels).float().mean()

        self.info.update({
        'loss': loss.detach().clone(),
        'i2t_acc': i2t_acc.detach().clone(),
        't2i_acc': t2i_acc.detach().clone(),
        })

        return loss, self.info

    # @gather_and_scale_wrapper
    # def forward(self, teacher_clean, student_clean, student_noisy, logit_scale=1.0):
    
    #     teacher_clean = teacher_clean.mean(dim=-1)  # [n, c]
    #     student_clean = student_clean.mean(dim=-1)
    #     student_noisy = student_noisy.mean(dim=-1)
       
    #     loss_1, info_1 = self.compute_pairwise_loss(teacher_clean, student_clean, logit_scale)
    #     loss_2, info_2 = self.compute_pairwise_loss(teacher_clean, student_noisy, logit_scale)
    #     loss = (loss_1 + loss_2) 

    #     self.info.update({
    #         'loss': loss.detach().clone(),
    #         'i2t_acc_clean': info_1['i2t_acc'],
    #         'i2t_acc_noisy': info_2['i2t_acc'],
    #         't2i_acc_clean': info_1['t2i_acc'],
    #         't2i_acc_noisy': info_2['t2i_acc'],
    #     })

    #     return loss, self.info
