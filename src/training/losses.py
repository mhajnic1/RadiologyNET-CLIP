import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedClipLoss(nn.Module):
    """CLIP contrastive loss that stops punishing genuinely-correct pairs.

    ClipLoss labels everything off the diagonal a negative, but images from the same
    exam carry byte-identical diagnosis text, so ~52% of batches of 64 contain at least
    one pair being pushed apart when it should not be. Worse, identical text embeds
    identically, so the softmax ends up with two indistinguishable logits and only one
    labelled correct, which is unsatisfiable and floors the loss at ln(m).

    Fix: drop those entries out of the softmax denominator entirely. The diagonal is
    never masked, so every row keeps at least one finite logit.
    """

    def forward(self, image_features, text_features, logit_scale, text_group):
        n = image_features.shape[0]
        device = image_features.device

        logits_per_image = logit_scale * image_features @ text_features.T

        same_text = text_group[:, None] == text_group[None, :]
        eye = torch.eye(n, dtype=torch.bool, device=device)
        false_negatives = same_text & ~eye

        logits_per_image = logits_per_image.masked_fill(false_negatives, float('-inf'))
        # the mask is symmetric, so the text->image direction reuses it transposed
        logits_per_text = logits_per_image.T

        labels = torch.arange(n, device=device)
        return (F.cross_entropy(logits_per_image, labels)
                + F.cross_entropy(logits_per_text, labels)) / 2


def count_false_negatives(text_group):
    n = text_group.shape[0]
    same = text_group[:, None] == text_group[None, :]
    return int((same.sum() - n) // 2)
