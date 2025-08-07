import datetime
import pandas as pd

from nfl_data_loader.api.sources.events.pbp.pbp import get_play_by_play
from nfl_data_loader.workflows.transforms.events.epa import make_rushing_epa, make_passing_epa
from nfl_data_loader.workflows.transforms.events.game import make_weekly_avg_group_features
from nfl_data_loader.workflows.transforms.events.penalty import make_avg_penalty_group_features
from nfl_data_loader.workflows.transforms.events.play import make_normal_play_group_features, make_general_group_features
from nfl_data_loader.workflows.transforms.events.score import make_score_feature, make_qtr_score_group_features
from nfl_data_loader.workflows.transforms.general.general import stat_collection


class TeamComponent:
    """
    Aggregates team-level features, including rolling stats and advanced metrics, for team-based analysis and modeling.
    """
    def __init__(self, load_seasons, season_type=None):
        """
        Initialize the TeamComponent with seasons and season type.
        """
        self.load_seasons = load_seasons
        self.season_type = season_type
        self.db = self.extract()
        self.df = self.run_pipeline()

    def extract(self):
        """
        Extracts all relevant team-level data for the given seasons.
        Returns a dictionary of DataFrames.
        """
        """
        Extracting play by play data, schedules, elo and weekly offensive and defensive player metrics (rolled up into total team metrics).
        Each of these data groups are extracted and loaded for the given seasons and filtered for the regular season
        :param load_seasons:
        :return:
        """
        print(f"    Loading play-by-play data {datetime.datetime.now()}")

        data = pd.concat([get_play_by_play(season, self.season_type) for season in self.load_seasons])
        #print(f"    Loading schedule data {datetime.datetime.now()}")

        print(f"    Loading offensive player weekly data {datetime.datetime.now()}")
        off_weekly = pd.concat([stat_collection(season, season_type=self.season_type, mode='team') for season in self.load_seasons])

        print(f"    Loading defensive player weekly data {datetime.datetime.now()}")
        def_weekly = pd.concat([stat_collection(season, season_type=self.season_type, mode='opponent') for season in self.load_seasons])
        return {
            'pbp': data,
            'team_stats': off_weekly,
            'opp_stats': def_weekly
        }

    def run_pipeline(self):
        """
        Main pipeline to process and merge team-level features.
        Returns a DataFrame of enriched team features.
        """
        epa = make_score_feature(self.db['pbp'])
        a = make_weekly_avg_group_features(self.db['team_stats'], self.db['opp_stats'])
        b = make_rushing_epa(self.db['pbp'])
        c = make_passing_epa(self.db['pbp'])
        d = make_avg_penalty_group_features(self.db['pbp'])
        e = make_normal_play_group_features(self.db['pbp'])
        f = make_general_group_features(self.db['pbp'])
        g = make_qtr_score_group_features(self.db['pbp'])

        groups = [
            a, b, c, d, e, f, g
        ]

        for group in groups:
            epa = pd.merge(epa, group, on=['team', 'season', 'week'], how='left')
        return epa.copy()