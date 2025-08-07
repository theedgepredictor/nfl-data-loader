import pandas as pd


def get_madden_ratings(season):
    try:
        df = pd.read_parquet(f'https://github.com/theedgepredictor/nfl-madden-data/raw/main/data/madden/dataset/{season}.parquet')
        return df
    except:
        return pd.DataFrame()