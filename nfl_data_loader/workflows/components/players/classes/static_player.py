from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class StaticPlayer:
    """
    Static player attributes containing identity, biographical details,
    draft information, and NFL Combine metrics that do not vary week-to-week.
    """
    # Identity fields
    player_id: str
    name: str
    first_name: str
    last_name: str
    pfr_id: Optional[str] = None
    espn_id: Optional[int] = None

    # Biographical details
    birth_date: Optional[datetime] = None
    height: Optional[int] = None  # in inches
    weight: Optional[int] = None  # in pounds
    headshot: Optional[str] = None

    # College information
    college_name: Optional[str] = None
    college_conference: Optional[str] = None

    # NFL career info
    rookie_season: Optional[int] = None

    # Draft information
    draft_year: Optional[int] = None
    draft_round: Optional[int] = None  # 0 if undrafted
    draft_pick: Optional[int] = None  # 0 if undrafted
    draft_team: Optional[str] = None

    # Combine metrics
    forty: Optional[float] = None  # 40-yard-dash time in seconds
    bench: Optional[int] = None  # Bench press reps at 225 lb
    vertical: Optional[float] = None  # Vertical jump height in inches
    broad_jump: Optional[int] = None  # Broad jump distance in inches
    cone: Optional[float] = None  # 3-cone-drill time in seconds
    shuttle: Optional[float] = None  # 20-yard shuttle time in seconds

    @property
    def height_formatted(self) -> str:
        """Returns height in feet and inches format (e.g., 6'2")"""
        if not self.height:
            return ""
        feet = self.height // 12
        inches = self.height % 12
        return f"{feet}'{inches}\""