from dataclasses import dataclass


@dataclass
class PlayerState:
    player_id: str
    game_id: int
    season: int
    week: int
    team: str
    high_pos_group: str
    position_group: str
    position: str
    starter: bool
    status: str
    report_status: str
    playerverse_status: str

    def __repr__(self):
        return f"Player({self.player_id}, {self.position}, {self.team}, W{self.week})"