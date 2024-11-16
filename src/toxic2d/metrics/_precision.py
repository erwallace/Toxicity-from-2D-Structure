def precision(labels, predicted):
    """
    Calculate precision.

    Args:
    predicted (torch.Tensor): Predicted values.
    labels (torch.Tensor): True values.

    Returns:
    float: Precision value.
    """
    true_positives = ((predicted == 1) & (labels == 1)).sum().item()
    false_positives = ((predicted == 1) & (labels == 0)).sum().item()

    if (true_positives + false_positives) == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)
