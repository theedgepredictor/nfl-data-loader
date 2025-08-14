import datetime
from typing import Dict

import pandas as pd

from nfl_data_loader.api.sources.players.adv.fantasy.projections import get_player_fantasy_projections
from nfl_data_loader.api.sources.players.adv.fantasy.ranks import get_fantasypros_ecr
from nfl_data_loader.api.sources.players.adv.fantasy.watson import get_player_watson_projections
from nfl_data_loader.utils.utils import find_year_for_season, find_week_for_season
from espn_api_orm.calendar.api import ESPNCalendarAPI


class PlayerFantasyComponent:
    """
    Main class for extracting, merging, and building weekly states for NFL players.
    Handles roster, starter, injury, depth chart, and game participation data.
    For PoC, focuses on QBs.
    """
    def __init__(self, load_seasons):
        self.load_seasons = load_seasons
        self.db = self.extract()
        self.df = self.run_pipeline()


    def extract(self) -> Dict[str, pd.DataFrame]:
        """
        Pulls raw dataframes:
          - projections_df: concat of get_player_fantasy_projections(season, mode='weekly', group='ALL')
          - watson_df:      concat of get_player_watson_projections(season)
          - ecr_df:         get_fantasypros_ecr(seasons)
        Returns dict of DataFrames with harmonized keys/dtypes.
        """
        print(f"    Loading fantasy projections data [{datetime.datetime.now()}]")
        projections = pd.concat(get_player_fantasy_projections(season, mode='weekly', group='ALL') for season in self.load_seasons)
        print(f"    Loading Watson projections data [{datetime.datetime.now()}]")
        watson = pd.concat(get_player_watson_projections(season) for season in self.load_seasons)

        print(f"    Loading FantasyPros ECR data [{datetime.datetime.now()}]")
        ecr = get_fantasypros_ecr(self.load_seasons)

        self.db = {
            "projections": projections,
            "watson": watson,
            "ecr": ecr,
        }
        return self.db

    def run_pipeline(self):
        df = pd.merge(self.db['projections'], self.db['watson'], on=['player_id','espn_id', 'season','week'], how='left')
        df = pd.merge(df, self.db['ecr'], on=['player_id','espn_id', 'season','week'], how='left')
        return df

    def run_all_boom_bust_candidates_for_evaluation(self):
        booms = []
        busts = []
        for season in self.df.season.unique():
            for week in range(1, (18 + 1 if season >= 2021 else 17 + 1)):
                boom_df, bust_df = self.generate_weekly_boom_bust_candidates(season, week)
                booms.append(boom_df)
                busts.append(bust_df)


    def run_latest_week_boom_bust_report(self):
        season = find_year_for_season()
        week = find_week_for_season()
        boom_df, bust_df = self.generate_weekly_boom_bust_candidates(season, week)

    def generate_weekly_boom_bust_candidates(self,season, week, n=10):
        df = self.df.copy()
        df = df[((df.season==season)&(df.week==week))].copy()

        pos_map = {
            'RB': 8,
            'WR': 7,
            'TE': 6,
            'D/ST': 5,
            'K': 5,
            'QB': 10
        }

        general_cols = [
            'espn_id',
            'player_id',
            'name',
            'position',
            'ecr',
            'player_owned_avg',
            'projected_points',
        ]

        suffix_cols = [
            'opponent_name',
            'opposition_rank',
            'data_timestamp'
        ]

        boom_dfs = []
        bust_dfs = []

        filter_check = df[((df.breakout_likelihood.notnull()) & (df.bust_likelihood.notnull()))].copy()
        if filter_check.shape[0] < 40:
            print('Not enough candidates for this week or missing data from source')
            return pd.DataFrame(), pd.DataFrame()

        for pos, projected_points_threshold in pos_map.items():
            filtered_df = df[((df.position == pos) & (df.projected_points >= projected_points_threshold))].copy().sort_values(['breakout_likelihood'], ascending=[False]).head(n)[general_cols + ['breakout_likelihood', 'projection_high_score'] + suffix_cols]
            boom_dfs.append(filtered_df)

        for pos, projected_points_threshold in pos_map.items():
            filtered_df = df[((df.position == pos) & (df.projected_points >= projected_points_threshold))].copy().sort_values(['bust_likelihood'], ascending=[False]).head(n)[general_cols + ['bust_likelihood', 'projection_low_score'] + suffix_cols]
            bust_dfs.append(filtered_df)

        boom_df = pd.concat(boom_dfs)
        bust_df = pd.concat(bust_dfs)
        return boom_df, bust_df

if __name__ == '__main__':
    pfc = PlayerFantasyComponent([2021,2022,2023,2024,2025])
    pfc.run_latest_week_boom_bust_report()