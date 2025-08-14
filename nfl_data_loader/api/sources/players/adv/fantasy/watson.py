import pandas as pd

from nfl_data_loader.api.sources.players.adv.fantasy.projections import T_ESPN_TO_PLAYERID
from nfl_data_loader.api.sources.players.general.players import get_player_ids
from nfl_data_loader.utils.formatters.reformat_team_name import team_id_repl



def get_player_watson_projections(season):
        df = pd.read_parquet(f'https://github.com/theedgepredictor/fantasy-data-pump/raw/main/processed/football/nfl/watson/{season}.parquet')
        df = team_id_repl(df)
        p_id = get_player_ids()
        p_id = p_id[p_id.espn_id.notnull()][['espn_id', 'gsis_id']]
        p_id.espn_id = p_id.espn_id.astype(int)
        p_id_dict = p_id.set_index('espn_id').to_dict()['gsis_id']
        df = df.rename(columns={'player_id': 'espn_id'})
        df['player_id'] = df['espn_id'].map({**T_ESPN_TO_PLAYERID, **p_id_dict})

        return df[[
            'player_id',
            'espn_id',
            #'position',
            'season',
            'week',
            'current_rank',
            'opponent_name',
            'opposition_rank',
            'is_on_injured_reserve',
            'is_suspended',
            'is_on_bye',
            'is_free_agent',
            #'projection_outside_projection',
            'projection_model_type',
            'projection_score',
            #'projection_score_distribution',
            'projection_distribution_name',
            'projection_low_score',
            'projection_high_score',
            'projection_simulation_projection',
            'breakout_likelihood',
            'bust_likelihood',
            'play_with_injury_likelihood',
            'play_without_injury_likelihood',
            'data_timestamp',
            'injury_status_date',
        ]].drop_duplicates(['player_id','espn_id','season','week'])