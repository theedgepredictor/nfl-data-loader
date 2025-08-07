from dataclasses import dataclass, field
from typing import List, Optional

from nfl_data_loader.workflows.components.players.classes.player_rating_state import PlayerRatingState
from nfl_data_loader.workflows.components.players.classes.static_player import StaticPlayer
from nfl_data_loader.workflows.components.players.classes.weekly_player_rating import WeeklyPlayerRating


@dataclass
class CareerPlayerRating(StaticPlayer):
    """Tracks a player's rating evolution over their career"""
    weekly_player_ratings: List[WeeklyPlayerRating] = field(default_factory=list)
    init_player_rating_state: PlayerRatingState = None
    init_season: Optional[int] = None
    init_week: Optional[int] = None
    last_updated_season: Optional[int] = None
    last_updated_week: Optional[int] = None

    @property
    def current_rating(self) -> Optional[WeeklyPlayerRating]:
        """Returns the player's most recent rating"""
        if not self.weekly_player_ratings:
            return None
        return self.weekly_player_ratings[-1]

    def needs_initialization(self) -> bool:
        """Check if player needs to be initialized"""
        return len(self.weekly_player_ratings) == 0
