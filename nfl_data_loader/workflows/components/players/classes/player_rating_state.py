from dataclasses import dataclass
from typing import Optional

@dataclass
class PlayerRatingState:
    """
    Player Rating State
    - Can be either a pre or post week rating state 
    - Used as that week's player rating that gets adjusted based on player position_group performance
    - Players core
    """
    player_id: str
    madden_id: Optional[str] = ""
    years_exp: Optional[int] = 0
    is_rookie: Optional[bool] = False
    rating: Optional[float] = 70.0
    last_season_av: Optional[int] = 4
    baseoverallrating: Optional[int] = 70
    agility: Optional[float] = 70
    acceleration: Optional[float] = 70
    speed: Optional[float] = 70
    stamina: Optional[float] = 70
    strength: Optional[float] = 70
    toughness: Optional[float] = 70
    injury: Optional[float] = 70
    awareness: Optional[float] = 70
    jumping: Optional[float] = 70
    trucking: Optional[float] = 0
    carrying: Optional[float] = 0
    ballcarriervision: Optional[float] = 0
    stiffarm: Optional[float] = 0
    spinmove: Optional[float] = 0
    jukemove: Optional[float] = 0
    throwpower: Optional[float] = 0
    throwaccuracyshort: Optional[float] = 0
    throwaccuracymid: Optional[float] = 0
    throwaccuracydeep: Optional[float] = 0
    playaction: Optional[float] = 0
    throwonrun: Optional[float] = 0
    catching: Optional[float] = 0
    shortrouterunning: Optional[float] = 0
    midrouterunning: Optional[float] = 0
    deeprouterunning: Optional[float] = 0
    spectacularcatch: Optional[float] = 0
    catchintraffic: Optional[float] = 0
    release: Optional[float] = 0
    runblocking: Optional[float] = 0
    passblocking: Optional[float] = 0

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