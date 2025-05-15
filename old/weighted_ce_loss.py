def weighted_cross_entropy_loss(outputs: torch.Tensor,
                                targets: torch.Tensor,
                                weights: torch.Tensor):
    """
    Calculates a custom weighted cross-entropy loss for a batch.
    Scales the outputs by the sum of the target probabilities (P(18plus)) for each sample. This is because the targets are soft labels comprising 4 out of 5 classes; the fifth class is 1 - P(18plus), where P(18plus) is the sum of the soft labels.
    Sample Loss = - sum_k ( target_k * log(output_k) )
    Batch Loss = Expected sample loss = sum_C ( P(C) * Loss(C) ) / sum_C ( P(C) ), where P(C) is the sample weight.

    Args:
        outputs (torch.Tensor): Model predictions (probabilities), shape [batch_size, num_classes].
        targets (torch.Tensor): Ground truth probabilities, shape [batch_size, num_classes].
        weights (torch.Tensor): Sample weights ('P(C)'), shape [batch_size, 1].

    Returns:
        torch.Tensor: Scalar tensor representing the weighted average loss.
    """
    # scale outputs by P(18plus) and clamp to avoid log(0)
    tots = targets.sum(dim=1, keepdim=True) # P(18plus) per sample
    outputs = outputs * tots
    outputs = torch.clamp(outputs, 1e-10, 1. - 1e-9)
    # Sample cross-entropy loss
    sample_loss = -torch.sum(targets * torch.log(outputs), dim=1, keepdim=True)
    weights_reshaped = weights.view_as(sample_loss)
    weighted_sample_losses = sample_loss * (weights.view_as(sample_loss))
    batch_loss = weighted_sample_losses.sum() / weights_reshaped.sum()
    return batch_loss