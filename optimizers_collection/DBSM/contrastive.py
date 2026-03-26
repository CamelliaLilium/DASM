import torch
import torch.nn.functional as F


def supervised_domain_contrastive_loss(features, domain_labels, temperature=0.07):
    """
    Supervised contrastive loss that treats domain IDs as classes.
    The implementation mirrors the formulation used in DAEF-VS/DVSF.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive.")

    if features.dim() != 2:
        raise ValueError("features must be 2-D (batch_size, dim).")

    device = features.device
    labels = domain_labels.to(device)
    batch_size = features.size(0)

    # Normalize features to stabilize cosine similarity
    features = F.normalize(features, dim=1)

    similarity_matrix = torch.div(
        torch.matmul(features, features.t()), temperature
    )

    # Mask to remove self-comparisons
    logits_mask = torch.ones_like(similarity_matrix) - torch.eye(
        batch_size, device=device
    )
    similarity_matrix = similarity_matrix * logits_mask

    label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    label_mask = label_mask * logits_mask

    positives_per_sample = label_mask.sum(dim=1)
    if torch.all(positives_per_sample == 0):
        return torch.tensor(0.0, device=device, requires_grad=True)

    exp_sim = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    mean_log_prob = (label_mask * log_prob).sum(dim=1) / positives_per_sample.clamp_min(1.0)

    loss = -mean_log_prob[positives_per_sample > 0].mean()
    return loss

