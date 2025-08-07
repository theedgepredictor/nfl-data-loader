import pandas as pd

from nfl_data_loader.schemas.players.position import POSITION_MAPPER
from nfl_data_loader.utils.formatters.reformat_team_name import team_id_repl


def collect_combine():
    data = pd.read_parquet("https://github.com/nflverse/nflverse-data/releases/download/combine/combine.parquet")
    data = team_id_repl(data)
    data['position_group'] = data.pos
    data.position_group = data.position_group.map(POSITION_MAPPER)
    return data.rename(columns={'player_name': 'name'})