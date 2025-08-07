import pandas as pd


def get_player_regular_season_game_fs(season, group='off'):
    try:
        df = pd.read_parquet(f'https://github.com/theedgepredictor/nfl-feature-store/raw/main/data/feature_store/player/{group}/regular_season_game/{season}.parquet')
        return df
    except:
        return pd.DataFrame()