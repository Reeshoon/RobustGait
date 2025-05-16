import torch
from torch.nn import functional as F

from .base import BaseLoss, gather_and_scale_wrapper

class KnowledgeDistillationLoss(BaseLoss):
    def __init__(self, tau=1.5, loss_type = "kl_divergence", loss_term_weight=1.0):
        super(KnowledgeDistillationLoss, self).__init__(loss_term_weight)
        self.loss_type = loss_type
        self.tau = tau

    def kl_divergence_loss(self, sil_feats, rgb_feats):
        """
        Computes KL Divergence loss with temperature scaling.
        """
        soft_targets = F.softmax(rgb_feats / self.tau, dim=1)
        soft_prob = F.log_softmax(sil_feats / self.tau, dim=1)
        kl_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size(0) * (self.tau ** 2)
        self.info.update({'kl_loss': kl_loss.detach().clone()})
        return kl_loss

    def cosine_similarity_loss(self, sil_feats, rgb_feats):
        """
        Computes cosine similarity loss.
        """
        cos_sim = F.cosine_similarity(sil_feats, rgb_feats, dim=-1)
        cosine_loss = (1 - cos_sim).mean()
        self.info.update({'cosine_loss': cosine_loss.detach().clone()})
        return cosine_loss

    def mse_loss(self, sil_feats, rgb_feats):
        """
        Computes Mean Squared Error (MSE) loss.
        """
        mse_loss = F.mse_loss(sil_feats, rgb_feats)
        self.info.update({'mse_loss': mse_loss.detach().clone()})
        return mse_loss

    def clip_loss(self, sil_feats, rgb_feats, temperature = 0.07):
        """
        Computes CLIP loss using cosine similarity and cross-entropy.
        """

        silhouette_embeddings = F.normalize(sil_feats, p=2, dim=-1)
        rgb_embeddings = F.normalize(rgb_feats, p=2, dim=-1)
        
        logits = torch.matmul(silhouette_embeddings, rgb_embeddings.T) / temperature
        targets = torch.arange(logits.shape[0]).to(logits.device)
        
        loss_sil_to_rgb = F.cross_entropy(logits, targets)
        loss_rgb_to_sil = F.cross_entropy(logits.T, targets)
        clip_loss = (loss_sil_to_rgb + loss_rgb_to_sil) / 2
        self.info.update({'clip_loss': clip_loss.detach().clone()})
        return clip_loss
    
    def clip_loss_all(self, sil_feats, rgb_feats, labels, temperature=0.07):
        """
        Computes symmetric CLIP loss using cosine similarity and cross-entropy,
        considering subject IDs to maximize within-subject similarity.
        """
        # Normalize the embeddings
        silhouette_embeddings = F.normalize(sil_feats, p=2, dim=-1)
        rgb_embeddings = F.normalize(rgb_feats, p=2, dim=-1)

        # Compute similarity matrix (logits)
        logits = torch.matmul(silhouette_embeddings, rgb_embeddings.T) / temperature

        # Create a subject-aware mask
        # same_subject_mask[i, j] = True if labels[i] == labels[j]
        same_subject_mask = labels.unsqueeze(0) == labels.unsqueeze(1)

        # Extract positive and negative logits
        positive_logits = logits[same_subject_mask]  # Similarities for same subjects
        negative_logits = logits[~same_subject_mask]  # Similarities for different subjects

        # Compute positive and negative losses for silhouette -> RGB
        positive_loss_sil_to_rgb = -torch.log(torch.sigmoid(positive_logits)).mean()  # Maximize positives
        negative_loss_sil_to_rgb = -torch.log(1 - torch.sigmoid(negative_logits)).mean()  # Minimize negatives
        loss_sil_to_rgb = positive_loss_sil_to_rgb + negative_loss_sil_to_rgb

        # Repeat for RGB -> silhouette (transpose logits)
        logits_t = logits.T
        positive_logits_t = logits_t[same_subject_mask]
        negative_logits_t = logits_t[~same_subject_mask]
        positive_loss_rgb_to_sil = -torch.log(torch.sigmoid(positive_logits_t)).mean()
        negative_loss_rgb_to_sil = -torch.log(1 - torch.sigmoid(negative_logits_t)).mean()
        loss_rgb_to_sil = positive_loss_rgb_to_sil + negative_loss_rgb_to_sil

        # Combine losses symmetrically
        clip_loss_all = (loss_sil_to_rgb + loss_rgb_to_sil) / 2

        # Store the loss for debugging purposes
        self.info.update({'clip_loss_all': clip_loss_all.detach().clone()})
        return clip_loss_all


    @gather_and_scale_wrapper
    def forward(self, sil_feats, rgb_feats, labels):
        """
        Computes the selected loss type.

        Args:
            sil_feats: Silhouette features.
            rgb_feats: RGB features.
            labels: Ground truth labels.
            loss_type: Type of loss to compute ('kl_divergence', 'cosine_similarity', 'mse', 'clip').
        """
        
        if self.loss_type == "kl_divergence":
            return self.kl_divergence_loss(sil_feats, rgb_feats), self.info
        elif self.loss_type == "cosine_similarity":
            return self.cosine_similarity_loss(sil_feats, rgb_feats), self.info
        elif self.loss_type == "mse":
            return self.mse_loss(sil_feats, rgb_feats), self.info
        elif self.loss_type == "clip":
            return self.clip_loss(sil_feats, rgb_feats, self.tau), self.info
        elif self.loss_type == "clip_all":
            return self.clip_loss(sil_feats, rgb_feats, labels, self.tau), self.info
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
