## Add from nfl-madden-data pump
import pandas as pd

## Add from nfl-madden-data pump
def get_approximate_value(season):
    try:
        df = pd.read_csv(f'https://github.com/theedgepredictor/nfl-madden-data/raw/main/data/pfr/approximate_value/{season}.csv')
        return df
    except:
        return pd.DataFrame()