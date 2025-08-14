import pandas as pd
from nfl_data_loader.api.sources.events.games.games import get_schedules, get_detailed_weeks_for_season
from nfl_data_loader.api.sources.players.adv.fantasy.projections import T_MERGENAME_TO_PLAYERID, T_ESPN_TO_PLAYERID
from nfl_data_loader.api.sources.players.general.players import get_player_ids

from nfl_data_loader.utils.utils import find_year_for_season


def get_fantasypros_ecr(seasons):
    url = "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_fpecr.parquet"
    df = pd.read_parquet(url)

    pages = [
        '/nfl/rankings/ppr-cheatsheets.php',  ### fantasy preseason cheatsheet
        '/nfl/rankings/ros-ppr-overall.php',  ### fantasy weekly rankings for the rest of the season
        '/nfl/rankings/idp-cheatsheets.php',  ### IDP preseason cheatsheet
        '/nfl/rankings/ros-idp-overall.php',  ### IDP weekly rankings for the rest of the season
    ]

    df = df[df.fp_page.isin(pages)].copy()
    df['scrape_date'] = pd.to_datetime(df['scrape_date'])
    df['season'] = None
    ## Get unique scrape dates
    date_season_map = {}
    for date in df['scrape_date'].dt.date.unique():
        date_season_map[date.strftime('%Y-%m-%d')] = find_year_for_season(date)

    df['scrape_date'] = df['scrape_date'].dt.strftime('%Y-%m-%d')
    df['season'] = df['scrape_date'].map(date_season_map)
    df = df[df['season'].isin(seasons)].copy()

    ## Determine week number for each scrape_date in the season (for weekly data)
    df['scrape_date'] = pd.to_datetime(df['scrape_date'])
    ## Remove entries in March, April, May as these are offseason and not relevant
    df = df[~df.scrape_date.dt.month.isin([3, 4, 5])].copy()

    df['week'] = None
    date_week_map = {}
    for season in seasons:

        detailed_weeks_map = get_detailed_weeks_for_season(season)
        for date in df['scrape_date'][df.season == season].copy().dt.date.unique():
            date_week_map[date.strftime('%Y-%m-%d')] = 0  # Default to week 0 if no match found (these are preweek entries so this is treated as preseason)
            for week in detailed_weeks_map:
                if week['start_date'] <= pd.Timestamp(date) <= week['end_date']:
                    date_week_map[date.strftime('%Y-%m-%d')] = week['week']
                    break
    df['scrape_date'] = df['scrape_date'].dt.strftime('%Y-%m-%d')
    df['week'] = df['scrape_date'].map(date_week_map)
    ## Preseason cheatsheet consensus
    df.loc[((df.fp_page.str.contains('cheatsheets')) & (df.week != 1)), 'week'] = 0
    ## Remove week 0 entries for weekly rankings as these are last seasons final rankings and not relevant
    df = df[~((df.fp_page.str.contains('overall')) & (df.week == 0))].copy()
    ## Save only the latest week 0 for each season as the preseason ranking
    preseason_df = df[(df.fp_page.str.contains('cheatsheets')) & (df.week == 0)].copy()
    preseason_df = preseason_df.sort_values(by=['season', 'scrape_date'], ascending=[True, False]).groupby(['season', 'id']).first().reset_index()
    weekly_df = df[~((df.fp_page.str.contains('cheatsheets')) & (df.week == 0))].copy()
    df = pd.concat([preseason_df, weekly_df], ignore_index=True)

    ids = get_player_ids()

    fp_join = ids[ids.fantasypros_id.notnull()][['fantasypros_id', 'gsis_id', 'espn_id']].rename(columns={'gsis_id': 'player_id'}).copy()
    fp_join['fantasypros_id'] = fp_join['fantasypros_id'].astype(int).astype(str)
    # fp_join = fp_join[fp_join.player_id.notnull()].copy()

    df = df[df['id'].notnull()].copy()
    df['id'] = df['id'].astype(int).astype(str)
    df = df.merge(fp_join, left_on='id', right_on='fantasypros_id', how='left').drop(columns=['fantasypros_id', 'id'])

    # Select and return relevant columns
    df = df[[
        'player_id',
        'espn_id',
        'mergename',
        'fp_page',
        'season',
        'week',
        'pos',
        'team',
        'ecr',
        'sd',
        'best',
        'worst',
        'player_owned_avg',
        'player_owned_espn',
        'player_owned_yahoo',
        'rank_delta',
        'scrape_date'
    ]].reset_index(drop=True).sort_values(by=['fp_page', 'season', 'ecr', 'week'], ascending=[False, False, True, True])
    team_df = df[df.pos == 'DST'].copy()
    team_df['player_id'] = team_df['mergename'].map(T_MERGENAME_TO_PLAYERID)
    team_df['espn_id'] = team_df['player_id'].map({v: k for k, v in T_ESPN_TO_PLAYERID.items()})
    player_df = df[df.pos != 'DST'].copy()
    player_df = player_df[((player_df['player_id'].notnull()) | (player_df['espn_id'].notnull()))].copy()
    df = pd.concat([player_df, team_df], ignore_index=True)
    df = df[df.espn_id.notnull()].copy().drop(columns=['team', 'pos'])
    df['espn_id'] = df['espn_id'].astype(int)
    if df.week.max() == 0:
        df['week'] = 1
    return df