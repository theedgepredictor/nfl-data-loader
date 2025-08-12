import pandas as pd

from nfl_data_loader.schemas.players.position import HIGH_POSITION_MAPPER, POSITION_MAPPER
from nfl_data_loader.schemas.players.espn_id_mapper import ESPN_ID_MAPPER
from nfl_data_loader.utils.formatters.reformat_team_name import team_id_repl

def get_player_ids():
    return pd.read_csv('https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv')

def collect_players():
    data = pd.read_parquet("https://github.com/nflverse/nflverse-data/releases/download/players_components/players.parquet")
    data = team_id_repl(data)
    data['position_group'] = data.position
    data.position_group = data.position_group.map(POSITION_MAPPER)
    data['high_pos_group'] = data.position_group
    data.high_pos_group = data.high_pos_group.map(HIGH_POSITION_MAPPER)
    data['status_abbr'] = data.status
    data.status_abbr = data.status_abbr.fillna('N')
    data.status_abbr = data.status_abbr.apply(lambda x: x[0])
    data.status_abbr = data.status_abbr.replace(['W', 'E', 'I', 'N'], ['N', 'N', 'N', 'N'])
    data = data.rename(columns={'display_name': 'name', 'gsis_id': 'player_id'})

    def add_missing_draft_data(df):
        ## load missing draft data ##
        missing_draft = pd.read_csv(
            'https://github.com/greerreNFL/nfeloqb/raw/refs/heads/main/nfeloqb/Manual%20Data/missing_draft_data.csv',
        )
        ## groupby id to ensure no dupes ##
        missing_draft = missing_draft.groupby(['player_id']).head(1)
        ## rename the cols, which will fill if main in NA ##
        missing_draft = missing_draft.rename(columns={
            'rookie_year': 'rookie_season_fill',
            'draft_number': 'draft_pick_fill',
            'entry_year': 'draft_year_fill',
            'birth_date': 'birth_date_fill',
        })
        ## add to data ##
        df = pd.merge(
            df,
            missing_draft[[
                'player_id', 'rookie_season_fill', 'draft_pick_fill',
                'draft_year_fill', 'birth_date_fill'
            ]],
            on=['player_id'],
            how='left'
        )
        ## fill in missing data ##
        for col in [
            'rookie_season', 'draft_pick', 'draft_year', 'birth_date'
        ]:
            ## fill in missing data ##
            df[col] = df[col].combine_first(df[col + '_fill'])
            ## and then drop fill col ##
            df = df.drop(columns=[col + '_fill'])
        ## return ##
        return df

    data = add_missing_draft_data(data)

    ids = pd.read_csv("https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv")
    id_map_df = ids[((ids.espn_id.notnull())&(ids.gsis_id.notnull()))][['espn_id','gsis_id']].copy()
    id_map_df['espn_id'] = id_map_df['espn_id'].astype(int).astype(str)
    gsis_to_espn_filler = dict(zip(id_map_df.gsis_id,id_map_df.espn_id))
    data['filled_espn_id'] = data.player_id
    data['filled_espn_id'] = data['filled_espn_id'].map({**ESPN_ID_MAPPER, **gsis_to_espn_filler})
    data['espn_id'] = data['espn_id'].fillna(data['filled_espn_id'])
    data = data.drop(columns=['filled_espn_id'])
    return data

if __name__ == '__main__':
    collect_players()
