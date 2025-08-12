import pandas as pd

from nfl_data_loader.utils.formatters.reformat_game_scores import score_clean
from nfl_data_loader.utils.formatters.reformat_team_name import team_id_repl


def get_schedules(seasons, season_type='REG'):
    if min(seasons) < 1999:
        raise ValueError('Data not available before 1999.')
    ## apply ##
    scheds = pd.read_csv('http://www.habitatring.com/games.csv')
    scheds = score_clean(scheds)
    scheds = scheds[scheds['season'].isin(seasons)].copy()
    if season_type == 'REG':
        scheds = scheds[scheds.game_type=='REG'].copy()
    scheds = team_id_repl(scheds)
    scheds['game_id'] = scheds['season'].astype(str) + '_' + scheds['week'].astype(str) + '_' + scheds['home_team'] + '_' + scheds['away_team']
    return scheds

def get_detailed_weeks_for_season(season):
    # Determine week number for each scrape_date (for weekly data)
    sched = get_schedules([season], season_type='ALL')[['week', 'gameday', 'gametime']]
    sched['datetime'] = pd.to_datetime(sched['gameday'] + ' ' + sched['gametime'])
    # Compute last game date per week
    first_game_of_weeks = sched.groupby('week')['datetime'].min().reset_index().rename(columns={'datetime': 'start_date'})

    end_date_of_weeks = first_game_of_weeks.copy()
    end_date_of_weeks['end_date'] = end_date_of_weeks['start_date'] + pd.Timedelta(days=6, hours=23, minutes=59, seconds=59)
    return end_date_of_weeks[['week', 'start_date', 'end_date']].to_dict('records')
