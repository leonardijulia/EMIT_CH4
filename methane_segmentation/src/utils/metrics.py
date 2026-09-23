import torch

def calculate_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculates Intersection over Union (IoU) for binary masks."""
    # Convert logits to binary predictions
    probs = torch.sigmoid(preds)
    preds_binary = (probs > threshold).float()
    targets_binary = targets.float()

    intersection = (preds_binary * targets_binary).sum().item()
    union = preds_binary.sum().item() + targets_binary.sum().item() - intersection
    
    if union == 0:
        return 1.0 if targets_binary.sum().item() == 0 else 0.0
    return intersection / union