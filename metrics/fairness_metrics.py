from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass
class FairnessMetrics:
      group: str
      true_positive_rate: float
      false_positive_rate: float
      true_negative_rate: float
      false_negative_rate: float
      positive_predictive_value: float
      selection_rate: float


@dataclass
class EqualizedOddsResult:
      tpr_diff: float
      fpr_diff: float
      equalized_odds_satisfied: bool
      group_metrics: Dict[str, FairnessMetrics]


def _group_confusion_metrics(
      y_true: np.ndarray, y_pred: np.ndarray
) -> FairnessMetrics:
      tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
      tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
      fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
      tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
      fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
      ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
      sr = (tp + fp) / (tp + fp + tn + fn)
      return FairnessMetrics(
          group="",
          true_positive_rate=tpr,
          false_positive_rate=fpr,
          true_negative_rate=tnr,
          false_negative_rate=fnr,
          positive_predictive_value=ppv,
          selection_rate=sr,
      )


def equalized_odds(
      df: pd.DataFrame,
      protected_col: str,
      label_col: str,
      prediction_col: str,
      threshold: float = 0.05,
) -> EqualizedOddsResult:
      group_metrics: Dict[str, FairnessMetrics] = {}
      for group, group_df in df.groupby(protected_col):
                metrics = _group_confusion_metrics(
                              group_df[label_col].values,
                              group_df[prediction_col].values,
                )
                metrics.group = str(group)
                group_metrics[str(group)] = metrics

      tpr_values = [m.true_positive_rate for m in group_metrics.values()]
      fpr_values = [m.false_positive_rate for m in group_metrics.values()]
      tpr_diff = max(tpr_values) - min(tpr_values)
      fpr_diff = max(fpr_values) - min(fpr_values)

    return EqualizedOddsResult(
              tpr_diff=tpr_diff,
              fpr_diff=fpr_diff,
              equalized_odds_satisfied=tpr_diff <= threshold and fpr_diff <= threshold,
              group_metrics=group_metrics,
    )


def demographic_parity(
      df: pd.DataFrame,
      protected_col: str,
      prediction_col: str,
) -> Dict[str, float]:
      return {
                str(group): float(group_df[prediction_col].mean())
                for group, group_df in df.groupby(protected_col)
      }


def individual_fairness_score(
      embeddings: np.ndarray,
      predictions: np.ndarray,
      k: int = 10,
) -> float:
      n = len(embeddings)
      consistency_scores = []
      for i in range(n):
                dists = np.linalg.norm(embeddings - embeddings[i], axis=1)
                dists[i] = np.inf
                neighbors = np.argsort(dists)[:k]
                neighbor_preds = predictions[neighbors]
                consistency_scores.append(
                    float(np.mean(neighbor_preds == predictions[i]))
                )
            return float(np.mean(consistency_scores))


def summarize_fairness(
      df: pd.DataFrame,
      protected_col: str,
      label_col: str,
      prediction_col: str,
) -> Dict[str, object]:
      eo = equalized_odds(df, protected_col, label_col, prediction_col)
    dp = demographic_parity(df, protected_col, prediction_col)
    max_dp = max(dp.values())
    min_dp = min(dp.values())

    return {
              "demographic_parity": dp,
              "parity_gap": max_dp - min_dp,
              "disparate_impact": min_dp / max_dp if max_dp > 0 else 0.0,
              "equalized_odds": {
                            "tpr_diff": eo.tpr_diff,
                            "fpr_diff": eo.fpr_diff,
                            "satisfied": eo.equalized_odds_satisfied,
              },
              "group_metrics": {
                            k: vars(v) for k, v in eo.group_metrics.items()
              },
    }
