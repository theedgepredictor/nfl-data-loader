from nfl_data_loader.schemas.players.position import POSITION_MAPPER, HIGH_POSITION_MAPPER
from nfl_data_loader.utils.formatters.reformat_team_name import team_id_repl


def collect_weekly_espn_player_stats(season, week=None, season_type=None,  group=''):
    if group in ['def', 'kicking']:
        group = '_' + group
    data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats{group}.parquet')
    if week is not None:
        data = data[((data.season < season) | ((data.season == season) & (data.week <= week)))].copy()
    else:
        data = data[data.season <= season].copy()
    if season_type is not None:
        data = data[((data.season_type == season_type))].copy()
    data = team_id_repl(data)
    data['position_group'] = data.position
    data.position_group = data.position_group.map(POSITION_MAPPER)
    data['high_pos_group'] = data.position_group
    data.high_pos_group = data.high_pos_group.map(HIGH_POSITION_MAPPER)
    data['status'] = 'ACT'
    return data