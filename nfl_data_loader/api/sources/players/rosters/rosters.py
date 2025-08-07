import numpy as np
import pandas as pd

from nfl_data_loader.api.sources.events.games.games import get_schedules
from nfl_data_loader.api.sources.players.general.players import collect_players
from nfl_data_loader.schemas.players.position import POSITION_MAPPER, HIGH_POSITION_MAPPER
from nfl_data_loader.utils.formatters.general import df_rename_fold
from nfl_data_loader.utils.formatters.reformat_team_name import team_id_repl


def _fill_pre_2002_roster(year):
    r_data = pd.read_parquet(f"https://github.com/nflverse/nflverse-data/releases/download/rosters/roster_{year}.parquet")
    r_data = r_data[[
        'gsis_id',
        'season',
        'team',
        'depth_chart_position',
        'position',
        'jersey_number',
        'status',
        'years_exp',
        'birth_date',
        'full_name'
    ]]
    rosters = []
    for week in range(1, 18 + 4):
        snapshot = r_data.copy()
        snapshot['week'] = week
        snapshot['status_description_abbr'] = snapshot['status']
        rosters.append(snapshot)
    return pd.concat(rosters, axis=0).reset_index(drop=True)

def collect_roster(year):
    try:
        if year < 2002:
            player_nfld_df = _fill_pre_2002_roster(year)
        else:
            player_nfld_df = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{year}.parquet')
    except Exception as e:
        if year < 2024:
            return pd.DataFrame()
        print(f'Cant get latest rosters for {year}...using latest player pull as week 1 data')
        player_nfld_df = collect_players()[['player_id', 'birth_date','position', 'latest_team','status_abbr', 'status','years_of_experience','jersey_number']]
        player_nfld_df = player_nfld_df.rename(
            columns={
                'latest_team': 'team',
                'years_of_experience': 'years_exp',
                'status_abbr': 'status_description_abbr',
                'player_id': 'gsis_id'
            })
        player_nfld_df['season'] = year
        player_nfld_df['week'] = 1

    player_nfld_df = team_id_repl(player_nfld_df)
    player_nfld_df = player_nfld_df[[
        'season',
        'week',
        'team',
        'position',
        'depth_chart_position',
        'jersey_number',
        'birth_date',
        'status',
        'status_description_abbr',
        'gsis_id',
        'full_name',
        # 'sportradar_id',
        # 'yahoo_id',
        # 'rotowire_id',
        # 'pff_id',
        # 'fantasy_data_id',
        # 'sleeper_id',
        'years_exp',
        # 'headshot_url',
        # 'ngs_position',
        # 'game_type',

        # 'football_name',
        # 'esb_id',
        # 'gsis_it_id',
        # 'smart_id',
    ]].rename(columns={'full_name': 'name', 'gsis_id': 'player_id'})
    player_nfld_df['jersey_number'] = player_nfld_df['jersey_number'].astype(str)
    player_nfld_df['jersey_number'] = player_nfld_df['jersey_number'].str.extract('(\d+)') # Only numeric jersey numbers
    player_nfld_df['jersey_number'] = player_nfld_df['jersey_number'].fillna(-1).astype(int) # Fill with -1 to avoid convert to float
    player_nfld_df['jersey_number'] = player_nfld_df['jersey_number'].astype(str) # Convert to string
    player_nfld_df['jersey_number'] = player_nfld_df['jersey_number'].replace("-1", np.nan) # Convert -1 to NaN

    player_nfld_df = player_nfld_df.loc[(
            (player_nfld_df.player_id.notnull()) & (player_nfld_df.birth_date.notnull()))].copy()
    player_nfld_df = player_nfld_df.drop(columns=['birth_date'])
    player_nfld_df = player_nfld_df.loc[player_nfld_df.player_id != ''].copy()
    player_nfld_df = player_nfld_df.rename(columns={'status_description_abbr': 'status_abbr'})
    player_nfld_df.status_abbr = player_nfld_df.status_abbr.fillna('N')
    player_nfld_df.status_abbr = player_nfld_df.status_abbr.apply(lambda x: x[0])
    player_nfld_df.status_abbr = player_nfld_df.status_abbr.replace(['W', 'E', 'I', 'N'], ['N', 'N', 'N', 'N'])
    player_nfld_df = player_nfld_df.reset_index().drop(columns='index')
    #player_nfld_df = player_nfld_df[player_nfld_df.week == 1].copy()
    player_nfld_df['position_group'] = player_nfld_df.position
    player_nfld_df.position_group = player_nfld_df.position_group.map(POSITION_MAPPER)
    player_nfld_df['high_pos_group'] = player_nfld_df.position_group
    player_nfld_df.high_pos_group = player_nfld_df.high_pos_group.map(HIGH_POSITION_MAPPER)
    return player_nfld_df

def get_starters(season):
    try:
        df = pd.read_parquet(f"https://github.com/theedgepredictor/event-data-pump/raw/main/rosters/football/nfl/{season}.parquet")
        df = df.rename(columns={'player_id': 'espn_id', 'team_abbr': 'team'})
        df = team_id_repl(df)

        # Join to events for season and week
        events_df = df_rename_fold(get_schedules([season], season_type=None), 'away_', 'home_')[
            ['game_id', 'season', 'game_type', 'week', 'team', 'espn']
        ].rename(columns={'espn': 'event_id'})

        #print(set(list(df.team.unique())).symmetric_difference(set(list(events_df.team.unique()))))
        df = events_df.merge(df, how='left', on=['event_id', 'team']).drop(columns=['event_id', 'period', 'active','team_id'])
        df['starter'] = df.starter.astype(bool)
        df['did_not_play'] = df.did_not_play.astype(bool)
        df = df[df.espn_id.notnull()].copy()
        df['espn_id'] = df['espn_id'].astype(int)
        df['espn_id'] = df['espn_id'].astype(str) # to match nflverse
        return df
    except Exception as e:
        print(e)
        return pd.DataFrame()

def collect_depth_chart(season):
    try:
        data = pd.read_parquet(f'https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.parquet')
        data = data.rename(columns={'club_code': 'team', 'depth_position': 'depth_chart_position', 'gsis_id': 'player_id'})
        data = team_id_repl(data)
        data = data[data.week.notnull()].copy()
        data.week = data.week.astype(int)
        data = data[[
            'season',
            'team',
            'week',
            'depth_team',
            'player_id',
            'position',
            'depth_chart_position',
        ]]
        data['position_group'] = data.position
        data.position_group = data.position_group.map(POSITION_MAPPER)
        data['depth_team'] = data['depth_team'].astype(int)
        return data
    except:
        return pd.DataFrame()