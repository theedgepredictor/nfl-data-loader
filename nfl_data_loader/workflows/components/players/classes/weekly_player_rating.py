from dataclasses import field, fields, dataclass
from typing import Dict, Sequence

import numpy as np

from nfl_data_loader.workflows.components.players.classes.player_rating_matrix import PlayerPositionGroupRatingMatrix
from nfl_data_loader.workflows.components.players.classes.player_rating_state import PlayerRatingState
from nfl_data_loader.workflows.components.players.classes.player_state import PlayerState

ATTRIBUTE_BUCKET = {
    "physical": {
        "speed", "acceleration", "agility", "strength", "jumping", "stamina",
    },
    "technical": {
        "throwpower", "throwaccuracyshort", "throwaccuracymid", "throwaccuracydeep",
        "ballcarriervision", "trucking", "carrying", "stiffarm", "spinmove", "jukemove",
        "catching", "spectacularcatch", "catchintraffic", "release", "shortrouterunning",
        "midrouterunning", "deeprouterunning",
    },
    "mental": {"awareness", "toughness", "injury"},
}

DECAY_BY_BUCKET = {"physical": 0.95, "technical": 0.85, "mental": 0.90}

# -----------------------------------------------------------
# 2. METRIC → ATTRIBUTE MAPS
# -----------------------------------------------------------

UNIVERSAL_METRIC_MAP: Dict[str, Sequence[str]] = {
    "epa": ["awareness"],
    "total_plays": ["stamina"],
    "total_turnovers": ["carrying", "awareness"],
}

POSITION_METRIC_MAP: Dict[str, Dict[str, Sequence[str]]] = {
    "QB": {
        "completion_percentage": ["throwaccuracyshort", "throwaccuracymid", "throwaccuracydeep"],
        "yards_per_pass_attempt": ["throwpower"],
        "interceptions": ["awareness"],
        "sack_rate": ["awareness"],
        "passing_epa": ["overallrating"],
    },
    "WR": {
        "yards_per_route": ["shortrouterunning", "midrouterunning", "deeprouterunning"],
        "catch_percentage": ["catching", "catchintraffic"],
        "target_share": ["release"],
        "receiving_yards_after_catch": ["jukemove", "spectacularcatch"],
    },
    "TE": {
        "yards_per_route": ["shortrouterunning", "midrouterunning", "deeprouterunning"],
        "catch_percentage": ["catching", "catchintraffic"],
        "target_share": ["release"],
        "receiving_yards_after_catch": ["jukemove", "spectacularcatch"],
    },
    "RB": {
        "yards_per_rush_attempt": ["ballcarriervision", "speed", "acceleration"],
        "broken_tackles": ["trucking", "stiffarm", "jukemove", "spinmove"],
        "rushing_fumbles_lost": ["carrying"],
        "rushing_epa": ["overallrating"],
    },
}

@dataclass
class WeeklyPlayerRating(PlayerState):
    pre_rating: PlayerRatingState
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    #team_metrics: Dict[str, float] = field(default_factory=dict)
    rating_matrix: PlayerPositionGroupRatingMatrix | None = None
    post_rating: PlayerRatingState | None = None

    MAX_WEEKLY_ADJ: int = 3
    RATING_TIERS = {
        'elite': (90, 99, 0.5),    # Range and max adjustment
        'great': (80, 89, 1.0),
        'good': (70, 79, 1.5),
        'average': (60, 69, 2.0),
        'below_average': (50, 59, 2.5),
        'poor': (0, 49, 3.0)
    }

    # ------------------------------------------------------------------
    def apply(self):
        adjustments = self._calculate_adjustments()
        self.post_rating = self._apply_adjustments(adjustments)

    # ------------------------------------------------------------------
    def _metric_factor(self, metric: str) -> float:
        val = self.performance_metrics.get(metric)
        if val is None or self.rating_matrix is None:
            return 0.0
        neg = metric in PlayerPositionGroupRatingMatrix.NEGATIVE_METRICS
        norm = self.rating_matrix._normalize_metric(metric, val)  # type: ignore
        return -norm if neg else norm

    def _calculate_adjustments(self) -> Dict[str, float]:
        perf_score = self.rating_matrix.calculate_rating() if self.rating_matrix else 0.0
        base_adj = np.sign(perf_score) * min(self.MAX_WEEKLY_ADJ, abs(perf_score) * 10)
        adj: Dict[str, float] = {}

        # Universal
        for m, attrs in UNIVERSAL_METRIC_MAP.items():
            if m not in self.performance_metrics:
                continue
            delta = base_adj * self._metric_factor(m)
            for a in attrs:
                adj[a] = adj.get(a, 0) + delta

        # Position specific
        pos_map = POSITION_METRIC_MAP.get(self.position, {})
        for m, attrs in pos_map.items():
            if m not in self.performance_metrics:
                continue
            delta = base_adj * self._metric_factor(m)
            for a in attrs:
                adj[a] = adj.get(a, 0) + delta
        return adj

    def _apply_adjustments(self, adj: Dict[str, float]) -> PlayerRatingState:
        new_vals = {f.name: getattr(self.pre_rating, f.name) for f in fields(self.pre_rating)}
        for attr, delta in adj.items():
            bucket = next((b for b, s in ATTRIBUTE_BUCKET.items() if attr in s), "technical")
            decay = DECAY_BY_BUCKET[bucket]
            adj_val = delta * (1 - decay)
            cur = getattr(self.pre_rating, attr)
            new_vals[attr] = int(np.clip(cur + adj_val, 0, 99))

        # Recalculate overall
        weights = self._position_weights()
        if weights:
            overall = sum(new_vals.get(a, 0) * w for a, w in weights.items()) / sum(weights.values())
            new_vals["overallrating"] = int(overall)

        return PlayerRatingState(**new_vals)

    def _position_weights(self) -> Dict[str, float]:
        pos = self.position
        if pos == "QB":
            return {
                "throwpower": 0.15,
                "throwaccuracyshort": 0.15,
                "throwaccuracymid": 0.15,
                "throwaccuracydeep": 0.15,
                "awareness": 0.15,
                "playaction": 0.10,
                "throwonrun": 0.10,
                "stamina": 0.05,
            }
        if pos in {"WR", "TE"}:
            return {
                "catching": 0.20,
                "shortrouterunning": 0.15,
                "midrouterunning": 0.15,
                "deeprouterunning": 0.15,
                "spectacularcatch": 0.10,
                "catchintraffic": 0.10,
                "release": 0.10,
                "speed": 0.05,
            }
        if pos == "RB":
            return {
                "ballcarriervision": 0.20,
                "speed": 0.15,
                "acceleration": 0.15,
                "agility": 0.15,
                "trucking": 0.10,
                "carrying": 0.10,
                "stiffarm": 0.05,
                "jukemove": 0.05,
                "spinmove": 0.05,
            }
        return {}