import numpy as np

class VAD_Metrics():
  def __init__(self) -> None:
    pass

  def _apply_tolerance(self, segments, tol_frames):
    expanded = []
    for start, end in segments:
        expanded.append((start - tol_frames, end + tol_frames))
    return expanded

  def _segments_to_mask(self, segments, total_frames):
    mask = np.zeros(total_frames, dtype=np.int32)

    for start, end in segments:
        mask[start:end] = 1
    return mask

  def _compute_confusion(self, gt_segments, pred_segments, tolerance_frames):
    """
    Результати TP, FP, FN, TN для одної пари gt_segments, pred_segments
    """

    total_frames = max(
        max(end for _, end in gt_segments),
        max(end for _, end in pred_segments)
    )

    if tolerance_frames > 0:
        gt_segments = self._apply_tolerance(gt_segments, tolerance_frames)

    gt_mask = self._segments_to_mask(gt_segments, total_frames)
    pred_mask = self._segments_to_mask(pred_segments, total_frames)

    TP = np.sum((gt_mask == 1) & (pred_mask == 1))
    FP = np.sum((gt_mask == 0) & (pred_mask == 1))
    FN = np.sum((gt_mask == 1) & (pred_mask == 0))
    TN = np.sum((gt_mask == 0) & (pred_mask == 0))

    return TP, FP, FN, TN


  def _compute_metrics_from_confusion(self, TP, FP, FN, TN):
    """
    Обраховує метрики {precision, recall, f1, FAR, miss_rate} на основі наданих TP, FP, FN, TN
    """
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    FAR = FP / (FP + TN + 1e-8)
    miss_rate = FN / (FN + TP + 1e-8)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "FAR": FAR,
        "miss_rate": miss_rate
    }

    return {k: round(float(v), 5) for k, v in metrics.items()}


  def _normalize_segments(self, segments):
    """
    segments: список dict-ів
    """
    filtered = []

    for seg in segments:
        filtered.append((seg["start"], seg["end"]))
    return filtered


  def compute_dataset_metrics(self, pairs_list, tolerance_frames=0):
    """
    pairs_list: список кортежів (pred_segments, gt_segments)
    """
    total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0

    for pred_segments, gt_segments in pairs_list:
        # _compute_confusion приймає готові сегменти
        TP, FP, FN, TN = self._compute_confusion(self._normalize_segments(gt_segments), pred_segments, tolerance_frames)

        total_TP += TP
        total_FP += FP
        total_FN += FN
        total_TN += TN

    return self._compute_metrics_from_confusion(total_TP, total_FP, total_FN, total_TN)
