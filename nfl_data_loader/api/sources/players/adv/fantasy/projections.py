import pandas as pd

from nfl_data_loader.api.sources.players.general.players import get_player_ids
from nfl_data_loader.utils.formatters.reformat_team_name import team_id_repl

T_ESPN_TO_PLAYERID = {
-16033: '00-BAL16033',
 -16006: '00-DAL16006',
 -16018: '00-NO16018',
 -16020: '00-NYJ16020',
 -16016: '00-MIN16016',
 -16007: '00-DEN16007',
 -16012: '00-KC16012',
 -16025: '00-SF16025',
 -16008: '00-DET16008',
 -16034: '00-HOU16034',
 -16027: '00-TB16027',
 -16002: '00-BUF16002',
 -16004: '00-CIN16004',
 -16028: '00-WAS16028',
 -16005: '00-CLE16005',
 -16001: '00-ATL16001',
 -16026: '00-SEA16026',
 -16030: '00-JAX16030',
 -16013: '00-LV16013',
 -16014: '00-LAR16014',
 -16019: '00-NYG16019',
 -16017: '00-NE16017',
 -16029: '00-CAR16029',
 -16024: '00-LAC16024',
 -16023: '00-PIT16023',
 -16015: '00-MIA16015',
 -16003: '00-CHI16003',
 -16009: '00-GB16009',
 -16021: '00-PHI16021',
 -16010: '00-TEN16010',
 -16022: '00-ARI16022',
 -16011: '00-IND16011'
}

T_MERGENAME_TO_PLAYERID = {
    "Baltimore Ravens":        "00-BAL16033",
    "Dallas Cowboys":          "00-DAL16006",
    "New Orleans Saints":      "00-NO16018",
    "New York Jets":           "00-NYJ16020",
    "Minnesota Vikings":       "00-MIN16016",
    "Denver Broncos":          "00-DEN16007",
    "Kansas City Chiefs":      "00-KC16012",
    "San Francisco 49ers":     "00-SF16025",
    "Detroit Lions":           "00-DET16008",
    "Houston Texans":          "00-HOU16034",
    "Tampa Bay Buccaneers":    "00-TB16027",
    "Buffalo Bills":           "00-BUF16002",
    "Cincinnati Bengals":      "00-CIN16004",
    "Washington Commanders":   "00-WAS16028",
    "Cleveland Browns":        "00-CLE16005",
    "Atlanta Falcons":         "00-ATL16001",
    "Seattle Seahawks":        "00-SEA16026",
    "Jacksonville Jaguars":    "00-JAX16030",
    "Las Vegas Raiders":       "00-LV16013",
    "Los Angeles Rams":        "00-LAR16014",
    "New York Giants":         "00-NYG16019",
    "New England Patriots":    "00-NE16017",
    "Carolina Panthers":       "00-CAR16029",
    "Los Angeles Chargers":    "00-LAC16024",
    "Pittsburgh Steelers":     "00-PIT16023",
    "Miami Dolphins":          "00-MIA16015",
    "Chicago Bears":           "00-CHI16003",
    "Green Bay Packers":       "00-GB16009",
    "Philadelphia Eagles":     "00-PHI16021",
    "Tennessee Titans":        "00-TEN16010",
    "Arizona Cardinals":       "00-ARI16022",
    "Indianapolis Colts":      "00-IND16011",
}

def get_player_fantasy_projections(season, mode='weekly', group='OFF'):
    """
    Fetches fantasy projections for players based on position group and timeframe.
    """
    try:
        df = pd.read_parquet(f'https://github.com/theedgepredictor/fantasy-data-pump/raw/main/processed/football/nfl/projections/{season}.parquet')
        df = team_id_repl(df)
        p_id = get_player_ids()
        p_id = p_id[p_id.espn_id.notnull()][['espn_id', 'gsis_id']]
        p_id.espn_id = p_id.espn_id.astype(int)
        p_id_dict = p_id.set_index('espn_id').to_dict()['gsis_id']
        df = df.rename(columns={'player_id': 'espn_id'})
        df['player_id'] = df['espn_id'].map({**T_ESPN_TO_PLAYERID, **p_id_dict})

        weekly_meta = [
            'season', 'week', 'player_id','espn_id', 'name', 'position', 'team',
            'percent_owned', 'percent_started', 'projected_points','PPR_draft_rank','STANDARD_draft_rank'
        ]

        season_meta = [
            'season', 'player_id','espn_id', 'name', 'position', 'team',
            'percent_owned', 'percent_started',
        ]

        offensive_cols = [
            'projected_rushing_attempts', 'projected_rushing_yards',
            'projected_rushing_touchdowns', 'projected_rushing2_pt_conversions',
            'projected_rushing40_plus_yard_td', 'projected_rushing50_plus_yard_td',
            'projected_rushing100_to199_yard_game', 'projected_rushing200_plus_yard_game',
            'projected_rushing_yards_per_attempt', 'projected_receiving_yards',
            'projected_receiving_touchdowns', 'projected_receiving2_pt_conversions',
            'projected_receiving40_plus_yard_td', 'projected_receiving50_plus_yard_td',
            'projected_receiving_receptions', 'projected_receiving100_to199_yard_game',
            'projected_receiving200_plus_yard_game', 'projected_receiving_targets',
            'projected_receiving_yards_per_reception', 'projected_2_pt_conversions',
            'projected_fumbles', 'projected_lost_fumbles', 'projected_turnovers',
            'projected_passing_attempts', 'projected_passing_completions',
            'projected_passing_yards', 'projected_passing_touchdowns',
            'projected_passing_interceptions', 'projected_passing_completion_percentage'
        ]

        defensive_cols = [
            'projected_defensive_solo_tackles',
            'projected_defensive_total_tackles',
            'projected_defensive_interceptions',
            'projected_defensive_fumbles',
            'projected_defensive_blocked_kicks',
            'projected_defensive_safeties',
            'projected_defensive_sacks',
            'projected_defensive_touchdowns',
            'projected_defensive_forced_fumbles',
            'projected_defensive_passes_defensed',
            'projected_defensive_assisted_tackles',
            'projected_defensive_points_allowed',
            'projected_defensive_yards_allowed',
            'projected_defensive0_points_allowed',
            'projected_defensive1_to6_points_allowed',
            'projected_defensive7_to13_points_allowed',
            'projected_defensive14_to17_points_allowed',
            'projected_defensive18_to21_points_allowed',
            'projected_defensive22_to27_points_allowed',
            'projected_defensive28_to34_points_allowed',
            'projected_defensive35_to45_points_allowed',
            'projected_defensive45_plus_points_allowed',
            'projected_defensive100_to199_yards_allowed',
            'projected_defensive200_to299_yards_allowed',
            'projected_defensive300_to349_yards_allowed',
            'projected_defensive350_to399_yards_allowed',
            'projected_defensive400_to449_yards_allowed',
            'projected_defensive450_to499_yards_allowed',
            'projected_defensive500_to549_yards_allowed',
            'projected_defensive550_plus_yards_allowed'
        ]

        special_teams_cols = [
            'projected_made_field_goals', 'projected_attempted_field_goals',
            'projected_missed_field_goals', 'projected_made_extra_points',
            'projected_attempted_extra_points', 'projected_missed_extra_points',
            'projected_kickoff_return_touchdowns', 'projected_kickoff_return_yards',
            'projected_punt_return_touchdowns', 'projected_punt_return_yards',
            'projected_punts_returned', 'projected_made_field_goals_from50_plus',
            'projected_attempted_field_goals_from50_plus',
            'projected_made_field_goals_from40_to49',
            'projected_attempted_field_goals_from40_to49',
            'projected_made_field_goals_from_under40',
            'projected_attempted_field_goals_from_under40'
        ]

        if group == 'OFF':
            potential_stat_cols = offensive_cols
            positions = ['QB', 'RB', 'WR', 'TE']
        elif group == 'DEF':
            potential_stat_cols = defensive_cols
            positions = ['D/ST']
        elif group == 'ST':
            potential_stat_cols = special_teams_cols
            positions = ['K']
        else:
            ## All
            potential_stat_cols = offensive_cols + defensive_cols + special_teams_cols
            positions = ['QB', 'RB', 'WR', 'TE', 'D/ST', 'K']

        # Only include stat columns that exist in the DataFrame
        stat_cols = [col for col in potential_stat_cols if col in df.columns]

        meta_cols = season_meta if mode == 'season' else weekly_meta
        all_cols = meta_cols + stat_cols
        df = df[all_cols]
        df = df[df.position.isin(positions)].copy()

        if mode == 'season':
            meta_df = df[meta_cols].drop_duplicates(['player_id'])
            ### need to make these raw
            meta_df['total_points'] = df.groupby('player_id')['points'].sum().reset_index(drop=True)
            meta_df['projected_total_points'] = df.groupby('player_id')['projected_points'].sum().reset_index(drop=True)
            meta_df['avg_points'] = df.groupby('player_id')['points'].mean().reset_index(drop=True)
            meta_df['projected_avg_points'] = df.groupby('player_id')['projected_points'].mean().reset_index(drop=True)
            stats_df = df[['player_id'] + stat_cols].groupby(['player_id']).sum()
            df = pd.merge(meta_df, stats_df, on=['player_id'])
        return df
    except Exception as e:
        print(f"Error fetching fantasy projections: {e}")
        return pd.DataFrame()