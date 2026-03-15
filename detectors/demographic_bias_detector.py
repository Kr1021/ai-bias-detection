from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GroupMetrics:
      group: str
      count: int
      positive_rate: float
      mean_score: float
      std_score: float


@dataclass
class BiasDetectionResult:
      protected_attribute: str
      group_metrics: List[GroupMetrics]
      demographic_parity_diff: float
      disparate_impact_ratio: float
      statistical_parity: float
      p_value: float
      is_biased: bool
      bias_severity: str
      affected_groups: List[str] = field(default_factory=list)


class DemographicBiasDetector:
      def __init__(
                self,
                parity_threshold: float = 0.1,
                impact_ratio_threshold: float = 0.8,
                significance_level: float = 0.05,
      ):
                self.parity_threshold = parity_threshold
                self.impact_ratio_threshold = impact_ratio_threshold
                self.significance_level = significance_level

      def _compute_group_metrics(
                self,
                df: pd.DataFrame,
                protected_col: str,
                score_col: str,
                outcome_col: str,
      ) -> List[GroupMetrics]:
                metrics = []
                for group, group_df in df.groupby(protected_col):
                              metrics.append(
                                                GroupMetrics(
                                                                      group=str(group),
                                                                      count=len(group_df),
                                                                      positive_rate=float(group_df[outcome_col].mean()),
                                                                      mean_score=float(group_df[score_col].mean()),
                                                                      std_score=float(group_df[score_col].std()),
                                                )
                              )
                          return metrics

    def _demographic_parity_difference(
              self, group_metrics: List[GroupMetrics]
    ) -> float:
              rates = [g.positive_rate for g in group_metrics]
              return max(rates) - min(rates)

    def _disparate_impact_ratio(
              self, group_metrics: List[GroupMetrics]
    ) -> float:
              rates = [g.positive_rate for g in group_metrics]
              if max(rates) == 0:
                            return 0.0
                        return min(rates) / max(rates)

    def _statistical_significance(
              self,
              df: pd.DataFrame,
              protected_col: str,
              score_col: str,
    ) -> float:
              groups = [
                  group_df[score_col].values
                  for _, group_df in df.groupby(protected_col)
    ]
        if len(groups) < 2:
                      return 1.0
                  _, p_value = stats.f_oneway(*groups)
        return float(p_value)

    def _severity(self, parity_diff: float, impact_ratio: float) -> str:
              if parity_diff > 0.2 or impact_ratio < 0.6:
                            return "HIGH"
elif parity_diff > 0.1 or impact_ratio < 0.8:
            return "MEDIUM"
        return "LOW"

    def detect(
              self,
              df: pd.DataFrame,
              protected_attribute: str,
              score_col: str,
              outcome_col: str,
    ) -> BiasDetectionResult:
              group_metrics = self._compute_group_metrics(
                  df, protected_attribute, score_col, outcome_col
    )
        parity_diff = self._demographic_parity_difference(group_metrics)
        impact_ratio = self._disparate_impact_ratio(group_metrics)
        p_value = self._statistical_significance(df, protected_attribute, score_col)

        overall_rate = df[outcome_col].mean()
        affected = [
                      g.group
                      for g in group_metrics
                      if abs(g.positive_rate - overall_rate) > self.parity_threshold
        ]

        is_biased = (
                      parity_diff > self.parity_threshold
                      or impact_ratio < self.impact_ratio_threshold
                      or p_value < self.significance_level
        )

        return BiasDetectionResult(
                      protected_attribute=protected_attribute,
                      group_metrics=group_metrics,
                      demographic_parity_diff=parity_diff,
                      disparate_impact_ratio=impact_ratio,
                      statistical_parity=1.0 - parity_diff,
                      p_value=p_value,
                      is_biased=is_biased,
                      bias_severity=self._severity(parity_diff, impact_ratio),
                      affected_groups=affected,
        )

    def detect_multiple(
              self,
              df: pd.DataFrame,
              protected_attributes: List[str],
              score_col: str,
              outcome_col: str,
    ) -> Dict[str, BiasDetectionResult]:
              return {
                  attr: self.detect(df, attr, score_col, outcome_col)
                            for attr in protected_attributes
              }
