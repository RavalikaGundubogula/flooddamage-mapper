from typing import Dict, Union
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def calculate_iou(
    pred: Union[np.ndarray, "torch.Tensor"],
    target: Union[np.ndarray, "torch.Tensor"],
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """
    Calculate Intersection over Union (Jaccard Index) for binary flood segmentation.
    IoU = TP / (TP + FP + FN + eps)
    """
    if TORCH_AVAILABLE and isinstance(pred, torch.Tensor):
        if pred.is_floating_point():
            pred_bin = (pred > threshold).float()
        else:
            pred_bin = (pred > 0).float()
        target_bin = (target > 0).float()

        intersection = (pred_bin * target_bin).sum().item()
        union = pred_bin.sum().item() + target_bin.sum().item() - intersection
        return float((intersection + eps) / (union + eps))

    # Numpy fallback
    p = np.asarray(pred)
    t = np.asarray(target)
    if np.issubdtype(p.dtype, np.floating):
        p_bin = (p > threshold).astype(np.float32)
    else:
        p_bin = (p > 0).astype(np.float32)
    t_bin = (t > 0).astype(np.float32)

    intersection = np.sum(p_bin * t_bin)
    union = np.sum(p_bin) + np.sum(t_bin) - intersection
    return float((intersection + eps) / (union + eps))


def calculate_dice(
    pred: Union[np.ndarray, "torch.Tensor"],
    target: Union[np.ndarray, "torch.Tensor"],
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """
    Calculate Dice Similarity Coefficient (F1-score) for binary flood segmentation.
    Dice = 2 * TP / (2 * TP + FP + FN + eps)
    """
    if TORCH_AVAILABLE and isinstance(pred, torch.Tensor):
        if pred.is_floating_point():
            pred_bin = (pred > threshold).float()
        else:
            pred_bin = (pred > 0).float()
        target_bin = (target > 0).float()

        intersection = (pred_bin * target_bin).sum().item()
        cardinality = pred_bin.sum().item() + target_bin.sum().item()
        return float((2.0 * intersection + eps) / (cardinality + eps))

    # Numpy fallback
    p = np.asarray(pred)
    t = np.asarray(target)
    if np.issubdtype(p.dtype, np.floating):
        p_bin = (p > threshold).astype(np.float32)
    else:
        p_bin = (p > 0).astype(np.float32)
    t_bin = (t > 0).astype(np.float32)

    intersection = np.sum(p_bin * t_bin)
    cardinality = np.sum(p_bin) + np.sum(t_bin)
    return float((2.0 * intersection + eps) / (cardinality + eps))


def calculate_precision(
    pred: Union[np.ndarray, "torch.Tensor"],
    target: Union[np.ndarray, "torch.Tensor"],
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """
    Calculate Precision: TP / (TP + FP + eps).
    Measures percentage of predicted flood pixels that are true flood pixels.
    """
    if TORCH_AVAILABLE and isinstance(pred, torch.Tensor):
        if pred.is_floating_point():
            pred_bin = (pred > threshold).float()
        else:
            pred_bin = (pred > 0).float()
        target_bin = (target > 0).float()

        tp = (pred_bin * target_bin).sum().item()
        tp_fp = pred_bin.sum().item()
        return float((tp + eps) / (tp_fp + eps))

    p = np.asarray(pred)
    t = np.asarray(target)
    p_bin = (p > threshold).astype(np.float32) if np.issubdtype(p.dtype, np.floating) else (p > 0).astype(np.float32)
    t_bin = (t > 0).astype(np.float32)

    tp = np.sum(p_bin * t_bin)
    tp_fp = np.sum(p_bin)
    return float((tp + eps) / (tp_fp + eps))


def calculate_recall(
    pred: Union[np.ndarray, "torch.Tensor"],
    target: Union[np.ndarray, "torch.Tensor"],
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """
    Calculate Recall / Sensitivity: TP / (TP + FN + eps).
    Measures percentage of actual flood pixels detected by the model.
    """
    if TORCH_AVAILABLE and isinstance(pred, torch.Tensor):
        if pred.is_floating_point():
            pred_bin = (pred > threshold).float()
        else:
            pred_bin = (pred > 0).float()
        target_bin = (target > 0).float()

        tp = (pred_bin * target_bin).sum().item()
        tp_fn = target_bin.sum().item()
        return float((tp + eps) / (tp_fn + eps))

    p = np.asarray(pred)
    t = np.asarray(target)
    p_bin = (p > threshold).astype(np.float32) if np.issubdtype(p.dtype, np.floating) else (p > 0).astype(np.float32)
    t_bin = (t > 0).astype(np.float32)

    tp = np.sum(p_bin * t_bin)
    tp_fn = np.sum(t_bin)
    return float((tp + eps) / (tp_fn + eps))


def evaluate_segmentation(
    pred: Union[np.ndarray, "torch.Tensor"],
    target: Union[np.ndarray, "torch.Tensor"],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate all semantic segmentation metrics.
    Returns dictionary with iou, dice, precision, recall rounded to 4 decimals.
    """
    iou = calculate_iou(pred, target, threshold=threshold)
    dice = calculate_dice(pred, target, threshold=threshold)
    prec = calculate_precision(pred, target, threshold=threshold)
    rec = calculate_recall(pred, target, threshold=threshold)

    return {
        "iou": round(float(iou), 4),
        "dice_score": round(float(dice), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
    }
