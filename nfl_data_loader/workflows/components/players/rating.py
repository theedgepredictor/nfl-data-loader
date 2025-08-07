import datetime
from typing import Dict, List

import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from nfl_data_loader.api.feature_stores.players.player_regular_season_game import get_player_regular_season_game_fs
from nfl_data_loader.workflows.components.players.classes.career_player_rating import CareerPlayerRating
from nfl_data_loader.workflows.components.players.classes.player_rating_matrix import QuarterbackPositionGroupRatingMatrix
from nfl_data_loader.workflows.components.players.classes.player_rating_state import PlayerRatingState
from nfl_data_loader.workflows.components.players.state import PlayerStateDataComponent
from nfl_data_loader.api.feature_stores.events.events import load_exavg_event_feature_store
from nfl_data_loader.workflows.transforms.players.player import get_preseason_players, MADDEN_FEATURES, impute_base_player_ratings, adjust_preseason_ratings


class PlayerRatingComponent:
    """
    Calculates weekly and overall ratings for players (PoC: QBs) using normalized stats, rolling averages, and regression to mean.
    Integrates Madden ratings and performance metrics.
    """

    def __init__(self, load_seasons, season_type=None):
        """
        Initializes the PlayerRatingComponent with seasons and season type.
        Loads data and runs the rating pipeline.
        """
        self.load_seasons = load_seasons
        self.season_type = season_type
        self.db = self.extract()
        # self.df = self.run_pipeline()
        self.career_player_ratings: Dict[str, CareerPlayerRating] = self.init_career_player_rating_map()
        self.player_ratings = None
        self.team_ratings = None

    def extract(self):
        """
        ### Input Data
        1. **Static Players**
        ```python
        'static_players': player_state_component.players
        ```
        - Contains unchanging player metadata
        - Used for joining against rating system
        - Provides player names and additional visualization data
        - Fields: player_id, name, birth_date, college, draft info, etc.

        2. **Player States**
        ```python
        'player_states': player_states
        ```
        - Weekly player state information
        - Contains merged weekly player data
        - Tracks player STATUS (PLAYED, INJURED, etc.)
        - Used to determine player availability and game participation

        3. **Offensive Weekly Player Stats**
        ```python
        'off_weekly_player_stats': off_players
        ```
        - Weekly offensive performance metrics
        - Contains game-level statistics
        - Used for rating calculations
        - Stats include: passing_yards, passing_tds, interceptions, completion_pct

        4. **Team Ratings**
        ```python
        'team_ratings': team_ratings
        ```
        - Team-level performance metrics
        - Used for contextualizing player performances
        - Loaded from event feature store

        5. **Preseason Players**
        ```python
        'preseason_players': preseason_players
        ```
        - Initial player ratings before season starts
        - Used for week 1 rating calculations
        - Includes Madden ratings and other baseline metrics
        """

        print(f"    Loading player data {datetime.datetime.now()}")
        player_state_component = PlayerStateDataComponent(self.load_seasons, season_type=self.season_type)
        player_states = player_state_component.run_pipeline()

        print(f"    Loading offensive aggregated player stat data {datetime.datetime.now()}")
        off_players = pd.concat([get_player_regular_season_game_fs(season, group='off') for season in self.load_seasons])

        print(f"    Loading event feature store {datetime.datetime.now()}")
        _, team_ratings = load_exavg_event_feature_store(self.load_seasons)

        print(f"    Loading preseason player rating data {datetime.datetime.now()}")
        preseason_players = pd.concat([get_preseason_players(season) for season in self.load_seasons])

        return {
            'static_players': player_state_component.players,
            'player_states': player_states,
            'off_weekly_player_stats': off_players,
            'team_ratings': team_ratings,
            'preseason_players': preseason_players,
        }

    def init_career_player_rating_map(self):
        """
        Initialize or update the career_player_ratings dictionary mapping player_ids to CareerPlayerRating objects.
        For refresh, starts at default init season-week; otherwise fills career player rating map up to week.

        Returns:
            Dict[str, CareerPlayerRating]: Updated career_player_ratings dictionary
        """
        # Initialize dictionary if not exists
        if not hasattr(self, 'career_player_ratings'):
            self.career_player_ratings = {}

        static_players_df = self.db['static_players'].copy()

        # Filter for QBs during PoC phase
        player_states_df = self.db['player_states'].copy()
        player_id_list = player_states_df[player_states_df.position_group.isin(['quarterback'])].copy().player_id.unique()
        static_players_df = static_players_df[static_players_df.player_id.isin(player_id_list)].copy()

        # Process each player's static data
        for _, row in static_players_df.iterrows():
            player_id = row.player_id

            career_player = CareerPlayerRating(
                # Identity fields
                player_id=row.player_id,
                name=row.name,
                first_name=row.first_name,
                last_name=row.last_name,
                pfr_id=row.pfr_id,
                espn_id=row.espn_id,

                # Biographical details
                birth_date=row.birth_date,
                height=row.height,
                weight=row.weight,
                headshot=row.headshot,

                # College information
                college_name=row.college_name,
                college_conference=row.college_conference,

                # NFL career info
                rookie_season=row.rookie_season,

                # Draft information
                draft_year=row.draft_year,
                draft_round=row.draft_round,
                draft_pick=row.draft_pick,
                draft_team=row.draft_team,

                # Combine metrics
                forty=row.forty,
                bench=row.bench,
                vertical=row.vertical,
                broad_jump=row.broad_jump,
                cone=row.cone,
                shuttle=row.shuttle,

                # Career rating specific fields
                init_season=None,
                init_week=None,
                weekly_player_ratings=[],
                last_updated_season=None,
                last_updated_week=None
            )

            # Add or update player in career_player_ratings dictionary
            self.career_player_ratings[player_id] = career_player
        return self.career_player_ratings

    def run_pipeline(self):
        """
        For each season and week:
        1. Initialize new players into the rating system
        2. Regress the player ratings
        3. Calculate weekly player ratings
            3a. Determine player availability
            3b. Calculate player adjustment matrix based on position group
            3c. Apply delta to player ratings to determine post_rating

        """
        for season in self.load_seasons:
            for week in range(1, (18 + 1 + 4 if season >= 2021 else 17 + 1 + 4)):

                player_states_df = self.db['player_states'].copy()
                player_states_df = player_states_df[((player_states_df.season == season) & (player_states_df.week == week))].copy()
                player_states_df = player_states_df[player_states_df.position_group.isin(['quarterback'])].copy()
                player_id_list = player_states_df.copy().player_id.unique()

                init_players_list = []
                regress_players_list = []
                for player_id in player_id_list:
                    ### Determine new players to the system
                    if self.career_player_ratings[player_id].init_player_rating_state is None:
                        init_players_list.append(player_id)
                    elif week == 1:  # Regress the player ratings
                        regress_players_list.append(player_id)
                    else:
                        pass  ## Good to go

                # Initialize new players into the rating system
                if len(init_players_list) > 0:
                    self.init_players(init_players_list, season, week)

                # Regress the player ratings
                if len(regress_players_list) > 0:
                    pass
                    # self.regress_players(regress_players_list, season, week)

                # Update player ratings for week
                for position_group in player_states_df.position_group.unique():
                    p_group = player_states_df[player_states_df.position_group == position_group].copy()
                    pre_ratings_df = []
                    for _, row in p_group.iterrows():
                        career_player = self.career_player_ratings[row.player_id]
                        pre_rating = career_player.init_player_rating_state if career_player.current_rating is None else career_player.current_rating.pre_rating
                        pre_ratings_df.append(pre_rating.__dict__)
                    pre_ratings_metrics = pd.DataFrame(pre_ratings_df)

                    adj_rating_metrics = self.position_based_adjustment_matrix(position_group, pre_ratings_metrics)
                    '''
                    for _, row in p_group.iterrows():
                        weekly_player_rating = WeeklyPlayerRating(
                            player_id=row['player_id'],
                            game_id=row['game_id'],
                            season=row['season'],
                            week=row['week'],
                            team=row['team'],
                            high_pos_group=row['high_pos_group'],
                            position_group=row['position_group'],
                            position=row['position'],
                            starter=row['starter'],
                            status=row['status'],
                            report_status=row.get('report_status', ''),
                            playerverse_status=row.get('playerverse_status', ''),
                            pre_rating=pre_rating,
                        )
                    '''

    def position_based_adjustment_matrix(self, position_group, pre_ratings_metrics):
        """
        Based on the position group apply the appropriate player stats and determine the adjustment matrix
        """
        if position_group == 'quarterback':
            schema_cols = QuarterbackPositionGroupRatingMatrix.get_schema_columns_for_metrics()
            season_cols = schema_cols['season_columns']
            form_cols = schema_cols['form_columns']
            adjustment_matrix_class = QuarterbackPositionGroupRatingMatrix(
                season_metrics=self.db['off_weekly_player_stats'][season_cols].copy(),
                form_metrics=self.db['off_weekly_player_stats'][form_cols].copy()
            )
        adjustment_matrix_class.adjust_metrics()
        adj_matrix = adjustment_matrix_class.compute_attribute_deltas(
            pre_ratings_metrics,
            season_weight=0.4
        )
        return adj_matrix

    def init_players(self, init_players_list, season, week):
        """Initialize new players into the rating system with 5-year lookback period"""
        # Get lookback seasons (up to 5 seasons back)
        lookback_seasons = range(max(1999, season - 5), season + 1)

        # Filter dataframes for lookback period
        player_states_df = self.db['player_states'].copy()
        player_states_df = player_states_df[
            ((player_states_df.season.isin(lookback_seasons)) &
             ((player_states_df.season < season) |
              ((player_states_df.season == season) & (player_states_df.week <= week))))
        ].copy()

        player_states_df = pd.merge(
            player_states_df,
            self.db['static_players'].copy(),
            how='left',
            on='player_id'
        )

        init_group = player_states_df[((player_states_df.season == season) & (player_states_df.week == week))].copy()
        init_group = init_group[init_group.player_id.isin(init_players_list)].copy()

        impute_group = player_states_df[~player_states_df.index.isin(init_group.index)].copy()
        impute_group = impute_group[((impute_group.week == 1))].copy()

        for position_group in init_group.position_group.unique():
            p_group = self.init_position_group(
                position_group,
                init_group[init_group.position_group == position_group].copy(),
                impute_group[impute_group.position_group == position_group].copy(),
                season,
                week
            )
            for _, row in p_group.iterrows():
                player_id = row.player_id
                init_player_rating_state = PlayerRatingState(
                    # Required field
                    player_id=row.player_id,

                    # Optional fields with defaults
                    madden_id=row.get('madden_id', ''),
                    years_exp=row.get('years_exp', 0),
                    is_rookie=row.get('is_rookie', False),
                    rating=row.get('rating', 70.0),
                    last_season_av=row.get('last_season_av', 4),
                    baseoverallrating=row.get('baseoverallrating', 70),

                    # Physical attributes
                    agility=row.get('agility', 70),
                    acceleration=row.get('acceleration', 70),
                    speed=row.get('speed', 70),
                    stamina=row.get('stamina', 70),
                    strength=row.get('strength', 70),
                    toughness=row.get('toughness', 70),
                    injury=row.get('injury', 70),
                    awareness=row.get('awareness', 70),
                    jumping=row.get('jumping', 70),

                    # Position-specific attributes
                    trucking=row.get('trucking', 0),
                    carrying=row.get('carrying', 0),
                    ballcarriervision=row.get('ballcarriervision', 0),
                    stiffarm=row.get('stiffarm', 0),
                    spinmove=row.get('spinmove', 0),
                    jukemove=row.get('jukemove', 0),

                    # QB-specific attributes
                    throwpower=row.get('throwpower', 70),
                    throwaccuracyshort=row.get('throwaccuracyshort', 70),
                    throwaccuracymid=row.get('throwaccuracymid', 70),
                    throwaccuracydeep=row.get('throwaccuracydeep', 70),
                    playaction=row.get('playaction', 70),
                    throwonrun=row.get('throwonrun', 70),

                    # Receiver attributes
                    catching=row.get('catching', 0),
                    shortrouterunning=row.get('shortrouterunning', 0),
                    midrouterunning=row.get('midrouterunning', 0),
                    deeprouterunning=row.get('deeprouterunning', 0),
                    spectacularcatch=row.get('spectacularcatch', 0),
                    catchintraffic=row.get('catchintraffic', 0),
                    release=row.get('release', 0),

                    # Blocking attributes
                    runblocking=row.get('runblocking', 0),
                    passblocking=row.get('passblocking', 0)
                )

                self.career_player_ratings[player_id].init_player_rating_state = init_player_rating_state
                self.career_player_ratings[player_id].init_season = season
                self.career_player_ratings[player_id].init_week = week

    def init_position_group(self, position_group, init_group, impute_group, season, week):
        lookback_seasons = range(max(1999, season - 5), season + 1)
        preseason_df = player_rating_component.db['preseason_players'].copy()
        preseason_df = preseason_df[
            ((preseason_df.season.isin(lookback_seasons)))
        ].copy()
        init_group = pd.merge(
            init_group,
            preseason_df.groupby('player_id').last().reset_index().drop(columns=['season']),
            how='left',
            on=['player_id']
        )

        impute_preseason_frames = []
        for season in impute_group.season.unique():
            p = preseason_df[preseason_df.season <= season].copy().groupby('player_id').last().reset_index().drop(columns=['season'])
            impute_preseason_frames.append(
                pd.merge(
                    impute_group[impute_group.season == season].copy(),
                    p,
                    how='left',
                    on=['player_id']
                )
            )
        impute_group = pd.concat(impute_preseason_frames, ignore_index=True)
        impute_group = impute_group[impute_group.baseoverallrating.notnull()].copy()
        if position_group == 'quarterback':
            init_group, impute_group, attributes = self._init_quarterbacks_group(init_group, impute_group, season, week)
        combined_data = pd.concat([init_group, impute_group], ignore_index=True)
        p_group = self.impute_base_player_ratings(combined_data, pos_group_features=attributes)
        p_group = p_group[((p_group.season == season) & (p_group.week == week)) & (p_group.player_id.isin(init_group.player_id.unique()))].copy()
        return p_group

    def _init_quarterbacks_group(self, init_group, impute_group, season, week):

        _QB_KPI_WEIGHTS = {
            # Efficiency Metrics (40% weight)
            'completion_percentage': 0.15,
            'yards_per_pass_attempt': 0.15,
            'passer_rating': 0.10,
            'VALUE_ELO': 0.10,
            'dakota': 0.10,

            # Production Metrics (30% weight)
            'passing_epa': 0.15,
            'passing_yards': 0.10,
            'passing_tds': 0.10,
            'passing_first_downs': 0.05,

            # Decision Making (30% weight)
            'touchdown_per_play': 0.15,
            'interceptions': -0.10,
            'sack_rate': -0.05
        }

        lookback_seasons = range(max(1999, season - 5), season + 1)
        attributes = [f"season_avg_{i}" for i in _QB_KPI_WEIGHTS.keys()]
        off_weekly_stats = player_rating_component.db['off_weekly_player_stats'].copy()
        off_weekly_stats = off_weekly_stats[
            ((off_weekly_stats.season.isin(lookback_seasons)) &
             ((off_weekly_stats.season < season) |
              ((off_weekly_stats.season == season) & (off_weekly_stats.week <= week))))
        ][['player_id', 'season', 'week'] + attributes].copy()

        # Add offensive stats to player data
        init_group = pd.merge(init_group, off_weekly_stats,
                              on=['player_id', 'season', 'week'], how='left')
        impute_group = pd.merge(impute_group, off_weekly_stats,
                                on=['player_id', 'season', 'week'], how='left')

        return init_group, impute_group, attributes

    def impute_base_player_ratings(self, df, pos_group_features):
        general_features = [
            'forty',
            'bench',
            'vertical',
            'broad_jump',
            'cone',
            'shuttle',
            'last_season_av',
        ]

        general_helper_features = [
            'height',
            'weight',
            # 'age',
            'years_exp',
            'draft_year',
            'draft_pick',
            'is_rookie'
        ]

        allowed_impute_cols = general_features + MADDEN_FEATURES + pos_group_features

        impute_df = df[allowed_impute_cols + general_helper_features]

        cols_with_missing = ((impute_df[allowed_impute_cols].isnull().sum() > 0) & (impute_df[allowed_impute_cols].isnull().sum() != df.shape[0])).reset_index().rename(columns={0: 'missing', 'index': 'col'})
        cols_with_missing = list(cols_with_missing[cols_with_missing['missing'] == True].col.values)

        df = df.drop(columns=cols_with_missing)

        impute_df = impute_df[cols_with_missing + general_helper_features].reset_index(drop=True)
        impute_df = impute_df.astype(float)

        imputer = IterativeImputer(
            random_state=0,
        )
        out = imputer.fit_transform(impute_df)
        return pd.concat([
            df.reset_index().drop(columns='index'),
            pd.DataFrame(data=out, columns=cols_with_missing + general_helper_features).drop(columns=general_helper_features)
        ], axis=1)

    def player_rating_season_pipeline(self, season, previous_rating_df):
        """
        Calculates ratings for all players in a season, using previous ratings for regression and rolling averages.
        Returns a DataFrame of season ratings.
        """
        if previous_rating_df.shape[0] != 0:
            p = previous_rating_df.sort_values(['player_id', 'season', 'week']).drop_duplicates(['player_id'], keep='last').rename(columns={
                'season': 'last_rating_season',
                'week': 'last_rating_week',
            }).copy()
        else:
            p = previous_rating_df
        season_ratings_df = []
        events_df = self.db['players'].drop_duplicates(['game_id'], keep='first')
        weeks = events_df[events_df.season == season].copy().week.unique()
        for week in list(weeks):
            p = self._weekly_player_pipeline(season, week, p)

    def _weekly_player_pipeline(self, season, week, previous_rating_df, position_group='quarterback'):
        """
        Calculates weekly ratings for QBs using normalized stats and combines them into a weighted rating.
        Applies regression to mean for preseason and rolling average for subsequent weeks.
        Returns a DataFrame of weekly QB ratings.
        """
        df = self.db['players']
        player_df = self.db['off_players']
        weekly_player_df = self.db['preseason_players'] if week == 1 else previous_rating_df

        df = df[((df['season'] == season) & (df.week == week))].copy()
        player_df = player_df[(
                (player_df['season'] < season - 1) | ((player_df['season'] == season) & (player_df.week == week))
        )].drop(columns=['position_group']).sort_values(['player_id', 'season', 'week']).drop_duplicates(['player_id'], keep='last').rename(columns={
            'season': 'last_stat_season',
            'week': 'last_stat_week',
        }).copy()

        weekly_player_df = pd.merge(weekly_player_df, df, on=['player_id', 'season'], how='left')
        weekly_player_df = weekly_player_df[weekly_player_df.position_group == position_group].copy()
        weekly_player_df = pd.merge(weekly_player_df, player_df, on=['player_id'], how='left')

        weekly_player_df['birth_date'] = pd.to_datetime(weekly_player_df.birth_date)
        weekly_player_df['birth_date'] = weekly_player_df['birth_date'] + pd.Timedelta(hours=4)
        weekly_player_df.birth_date = pd.to_datetime(weekly_player_df.birth_date, utc=True)
        weekly_player_df['age'] = (weekly_player_df['datetime'] - weekly_player_df['birth_date']).dt.days / 365

        # Clean up columns
        weekly_player_df = weekly_player_df.drop(columns=[
            'common_first_name',
            'first_name',
            'last_name',
            'short_name',
            'football_name',
            'suffix',
            'esb_id',
            'nfl_id',
            'pff_id',
            'otc_id',
            'espn_id',
            'smart_id',
            'headshot',
            'college_name',
            'college_conference',
            'rookie_season',
            'draft_team'
        ], errors='ignore')
        weekly_player_df = weekly_player_df.drop_duplicates(['player_id', 'season'])

        # --- QB Weekly Rating Calculation ---
        # Only keep QBs who played or were starters
        qbs = weekly_player_df.copy()
        # Example stat columns: passing_yards, passing_tds, interceptions, completion_pct
        stat_cols = ['passing_yards', 'passing_tds', 'interceptions', 'completion_pct']
        # Normalize stats (z-score)
        for col in stat_cols:
            if col in qbs.columns:
                mean = qbs[col].mean()
                std = qbs[col].std()
                if std > 0:
                    qbs[f'{col}_norm'] = (qbs[col] - mean) / std
                else:
                    qbs[f'{col}_norm'] = 0
            else:
                qbs[f'{col}_norm'] = 0

        # Weighted sum for weekly rating
        qbs['weekly_rating'] = (
                qbs['passing_yards_norm'] * 0.3 +
                qbs['passing_tds_norm'] * 0.4 -
                qbs['interceptions_norm'] * 0.2 +
                qbs['completion_pct_norm'] * 0.1
        )

        # Update overall rating (rolling average or regression)
        if week == 1:
            # Apply regression to mean for preseason
            def rating_regression(preseason_players_df, previous_rating_df):
                if previous_rating_df.shape[0] == 0:
                    return preseason_players_df
                # Simple regression: new_rating = mean + (old_rating - mean) * 2/3
                mean_rating = previous_rating_df['overallrating'].mean() if 'overallrating' in previous_rating_df.columns else 70
                preseason_players_df['overallrating'] = mean_rating + (preseason_players_df['overallrating'] - mean_rating) * (2 / 3)
                return preseason_players_df

            qbs = rating_regression(qbs, previous_rating_df)
        else:
            # Rolling average: combine previous and current week
            if previous_rating_df is not None and 'overallrating' in previous_rating_df.columns:
                prev = previous_rating_df[['player_id', 'overallrating']].set_index('player_id')
                qbs['overallrating'] = qbs.apply(
                    lambda row: (row['weekly_rating'] + prev.loc[row['player_id'], 'overallrating']) / 2 if row['player_id'] in prev.index else row['weekly_rating'],
                    axis=1
                )
            else:
                qbs['overallrating'] = qbs['weekly_rating']

        # Impute Missing Base Ratings
        qbs = impute_base_player_ratings(qbs)
        qbs = qbs.sort_values(['overallrating'], ascending=False)

        # Adjust preseason ratings if needed
        if week == 1:
            qbs = adjust_preseason_ratings(qbs)

        return qbs


if __name__ == '__main__':
    player_rating_component = PlayerRatingComponent([2004, 2005], season_type='REG')
    df = player_rating_component.run_pipeline()