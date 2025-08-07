from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Literal, List

import numpy as np
import pandas as pd


@dataclass
class PlayerPositionGroupRatingMatrix:
    """
    Base rating-matrix for a position-group (QB, RB, LB, …)

    Parameters
    ----------
    position_group : str
        Group name (e.g. 'quarterback').
    validation_requirement : Dict[str, int]
        Minimum per-player thresholds for *raw* season/form stats
        (e.g. {'pass_attempts': 5} for QBs).
    adjustment_fields : Dict[str, Direction]
        Mapping <metric_name> → 'positive' | 'negative'.
        • 'positive': higher metric ⇒ better player  
        • 'negative': lower metric  ⇒ better player
    season_metrics / form_metrics : pd.DataFrame
        Season-to-date averages & 5-game rolling forms for the group.
    """
    position_group: str
    validation_requirement: Dict[str, int] = field(default_factory=dict)
    adjustment_fields: Dict[str, str] = field(default_factory=dict)
    rating_attribute_stat_map: Dict[str, List[str]] = field(default_factory=dict)
    season_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    form_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #
    def adjust_metrics(self) -> None:
        """
        1.  Enforces validation thresholds (drops rows < threshold).
        2.  Converts *all* `adjustment_fields` into **direction-corrected
            z-scores** in both season & form DataFrames.
            · z = (x − mean) / std  
            · If metric is 'negative', multiply z by **−1** so that
              “higher-is-better” for every adjusted column.
        """
        self._apply_validation()
        self._zscore_adjustment(self.season_metrics,  prefix="season_avg_")
        self._zscore_adjustment(self.form_metrics,    prefix="form_")

    def compute_attribute_deltas(
        self,
        pre_ratings_df: pd.DataFrame,
        *,
        season_weight: float = 0.4,      # 0.4 season  + 0.6 form  → 1.0
    ) -> pd.DataFrame:
        """
        Returns the final Δ_z adjustment matrix used to update Madden
        attributes.

        Parameters
        ----------
        pre_ratings_df : DataFrame   – Madden attributes (one row / player_id)
        season_weight   : float      – Weight for season-average stats.
                                    Form weight = (1 − season_weight).

        Notes
        -----
        · `adjust_metrics()` **must** be called first so that both
        season_metrics & form_metrics are already z-scored.
        """
        season_delta = self._compute_attribute_deltas(
            pre_ratings_df, which="season"
        )
        form_delta   = self._compute_attribute_deltas(
            pre_ratings_df, which="form"
        )

        w_season = float(season_weight)
        w_form   = 1.0 - w_season

        combined = w_season * season_delta + w_form * form_delta
        return combined


    # ------------------------------------------------------------------ #
    # INTERNAL HELPERS
    # ------------------------------------------------------------------ #
    def _apply_validation(self) -> None:
        for raw_field, min_val in self.validation_requirement.items():
            for prefix, df in [("season_avg_", self.season_metrics),
                               ("form_",        self.form_metrics)]:
                col = f"{prefix}{raw_field}"
                if col in df.columns:
                    df.drop(index=df[df[col] < min_val].index, inplace=True)

    def _zscore_adjustment(self, df: pd.DataFrame, *, prefix: str) -> None:
        """
        Performs in-place z-scoring on df for every metric in
        `self.adjustment_fields`.  Handles std = 0 gracefully.
        """
        for metric, direction in self.adjustment_fields.items():
            col = f"{prefix}{metric}"
            if col not in df.columns:
                continue
            mean, std = df[col].mean(), df[col].std(ddof=0)
            if std == 0 or np.isnan(std):          # all players identical → 0.0
                z = pd.Series(0.0, index=df.index)
            else:
                z = (df[col] - mean) / std
            if direction == "negative":        # lower is better → flip sign
                z = -z
            df[col] = z            # overwrite with direction-corrected z-score

    def _compute_attribute_deltas(
        self,
        pre_ratings_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        pre_ratings_df : DataFrame
            One row per player with Madden attributes (e.g. throwpower, awareness…)

        Returns
        -------
        DataFrame
            index = player_id  
            columns = each Madden attribute in _ATTR_STAT_MAP  
            values = Δ_z (stat_z  –  rating_z)
        """
        # --- z-score Madden attributes --------------------
        attrs = list(self.rating_attribute_stat_map.keys())
        attr_z = pre_ratings_df.set_index("player_id")[attrs].apply(
            lambda col: (col - col.mean()) / (col.std(ddof=0) or 1)
        )

        # --- ensure season_metrics already adjusted -------
        # (caller must have run self.adjust_metrics())
        season = self.season_metrics.set_index("player_id")

        # --- build Δ_z matrix -----------------------------
        deltas = pd.DataFrame(index=season.index, columns=attrs, dtype=float)
        for attr, stat_list in self.rating_attribute_stat_map.items():
            # average z of mapped stats (they are already direction-corrected)
            stat_cols = [f"season_avg_{m}" for m in stat_list if f"season_avg_{m}" in season.columns]
            if not stat_cols:
                continue
            stat_z = season[stat_cols].mean(axis=1)
            deltas[attr] = stat_z - attr_z[attr]

        return deltas
    
@dataclass
class QuarterbackPositionGroupRatingMatrix(PlayerPositionGroupRatingMatrix):
    _QB_adjustment_directions = {
        # Efficiency
        'completion_percentage': 'positive',
        'yards_per_pass_attempt': 'positive',
        'passer_rating': 'positive',
        'VALUE_ELO': 'positive',
        'dakota': 'positive',
        # Production
        'passing_epa': 'positive',
        'passing_yards': 'positive',
        'passing_tds': 'positive',
        'passing_first_downs': 'positive',
        # Decision-making
        'touchdown_per_play': 'positive',
        'interceptions': 'negative',   # ↓ INT rate is good
        'sack_rate': 'negative'        # ↓ Sack % is good
    }

    _ATTR_STAT_MAP = {
        "throwpower":        ["passing_epa", "yards_per_pass_attempt", "touchdown_per_play"],
        "throwaccuracyshort":["completion_percentage"],
        "throwaccuracymid":  ["completion_percentage"],
        "throwaccuracydeep": ["yards_per_pass_attempt"],
        "awareness":         ["interceptions", "sack_rate"],
        "stamina":           ["total_plays"],           # total_plays must be present in season_metrics
    }

    def __init__(self,
                 season_metrics: pd.DataFrame,
                 form_metrics: pd.DataFrame):
        super().__init__(
            position_group='quarterback',
            validation_requirement={'pass_attempts': 5},
            adjustment_fields=self._QB_adjustment_directions,
            rating_attribute_stat_map = self._ATTR_STAT_MAP,
            season_metrics=season_metrics,
            form_metrics=form_metrics,
        )
    @classmethod
    def get_kpi_weights(cls) -> Dict[str, float]:
        """
        Get the KPI weights for quarterback rating calculations.
        
        Returns:
            Dict[str, float]: Dictionary of KPI metrics and their weights
        """
        return cls._QB_adjustment_directions
    
    @classmethod
    def get_schema_columns_for_metrics(cls):
        return {
            'season_columns': ['player_id']+[f'season_avg_{metric}' for metric in cls._QB_adjustment_directions.keys() + cls.validation_requirement.keys()],
            'form_columns': ['player_id']+[f'form_{metric}' for metric in cls._QB_adjustment_directions.keys() + cls.validation_requirement.keys()]
        }