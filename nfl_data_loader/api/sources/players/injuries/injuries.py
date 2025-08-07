import pandas as pd

from nfl_data_loader.schemas.players.position import POSITION_MAPPER
from nfl_data_loader.utils.formatters.reformat_team_name import team_id_repl


def collect_injuries(season):
    try:
        data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet')
        data = team_id_repl(data)
        data['position_group'] = data.position
        data.position_group = data.position_group.map(POSITION_MAPPER)
        return data.rename(columns={'gsis_id': 'player_id'})
    except Exception as e:
        return pd.DataFrame()
