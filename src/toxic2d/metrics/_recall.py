def recall(labels, predicted):
    """
    Calculate recall.

    Args:
    predicted (torch.Tensor): Predicted values.
    labels (torch.Tensor): True values.

    Returns:
    float: Recall value.
    """
    true_positives = ((predicted == 1) & (labels == 1)).sum().item()
    false_negatives = ((predicted == 0) & (labels == 1)).sum().item()

    if (true_positives + false_negatives) == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)
