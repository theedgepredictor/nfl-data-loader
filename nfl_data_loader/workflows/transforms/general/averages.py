import numpy as np
import pandas as pd


def dynamic_window_ewma(x):
    """
    Calculate rolling exponentially weighted features with a dynamic window size
    """
    values = np.zeros(len(x))
    for i, (_, row) in enumerate(x.iterrows()):
        epa = x.epa_shifted[:i + 1]
        if row.week > 10:
            values[i] = epa.ewm(min_periods=1, span=row.week).mean().values[-1]
        else:
            values[i] = epa.ewm(min_periods=1, span=10).mean().values[-1]

    return pd.Series(values, index=x.index)


def dynamic_window_rolling_average(x, attr, mode='season_avg'):
    """
    Calculate rolling features with a dynamic window size for the specified attribute.

    Parameters:
        x (DataFrame): DataFrame containing the play-by-play data grouped by team.
        attr (str): The attribute for which rolling average is calculated.
        mode (str, optional): The mode of the rolling average. Default is 'season_avg'.

    Returns:
        pd.Series: Series with the dynamic rolling for the attribute.
    """
    values = np.zeros(len(x))
    attr_shifted = f'{attr}_shifted'

    for i, (_, row) in enumerate(x.iterrows()):
        attr_data = x[attr_shifted][:i + 1]
        if mode == 'career_avg':
            values[i] = attr_data.mean()
        elif mode == 'season_avg':
            if row['week'] != 1:
                values[i] = attr_data.rolling(min_periods=1, window=row['week'] - 1).mean().values[-1]
            else:
                # Handle edge case for the first week or season start
                values[i] = attr_data.rolling(min_periods=1, window=18 if row['season'] >= 2021 else 17).mean().values[-1]
        elif mode == 'season_total':
            if row['week'] != 1:
                values[i] = attr_data.rolling(min_periods=1, window=row['week'] - 1).sum().values[-1]
            else:
                # Handle edge case for the first week or season start
                values[i] = attr_data.rolling(min_periods=1, window=18 if row['season'] >= 2021 else 17).sum().values[-1]
        elif mode == 'form':
            ### last 3 divided by career avg
            values[i] = attr_data.rolling(min_periods=1, window=3).mean().values[-1]
        else:
            values[i] = attr_data.rolling(min_periods=1, window=3).mean().values[-1]

    return pd.Series(values, index=x.index)

def ensure_sorted_index(df):
    # sort and put keys in the index for easier aligned ops
    df = df.sort_values(['player_id','season','week']).copy()
    return df.set_index(['player_id','season','week'])

def _shift_group(df_wide, cols):
    # shift all target cols by player
    g = df_wide.groupby(level=0, sort=False)  # level 0 -> player_id
    shifted = g[cols].shift()
    return shifted

def _within_season_expanding_sum_mean(shifted, how='mean'):
    """
    Vectorized within-season expanding over SHIFTED values:
    mean = csum_nonnull / ccount_nonnull (ignores NaNs),
    sum  = csum_nonnull.
    """
    # make a same-index Season grouper: (player_id, season)
    idx = shifted.index
    ps_grouper = [idx.get_level_values(0), idx.get_level_values(1)]  # player_id, season

    # non-null mask for counts
    nn = shifted.notna().astype(np.int64)

    # cumsum over season for sums and counts
    csum = shifted.fillna(0).groupby(ps_grouper, sort=False).cumsum()
    ccnt = nn.groupby(ps_grouper, sort=False).cumsum()

    if how == 'sum':
        out = csum
    else:  # mean
        out = csum / ccnt
        out = out.where(ccnt > 0)  # keep NaN if no prior data
    return out

def _career_rolling_lastN(shifted, N):
    """
    Player-level rolling over SHIFTED values for all columns at once.
    N can be scalar (int) -> one rolling,
    or an array per row to pick between two precomputed windows (17 vs 18).
    """
    g = shifted.groupby(level=0, sort=False)  # by player_id
    if np.isscalar(N):
        return g[shifted.columns].rolling(window=N, min_periods=1).mean() \
                .reset_index(level=0, drop=True)
    else:
        # We precompute both 17 and 18 and select by position later.
        roll17 = g[shifted.columns].rolling(window=17, min_periods=1).mean() \
                    .reset_index(level=0, drop=True)
        roll18 = g[shifted.columns].rolling(window=18, min_periods=1).mean() \
                    .reset_index(level=0, drop=True)
        return roll17, roll18

def _career_rolling_lastN_sum(shifted, N):
    g = shifted.groupby(level=0, sort=False)
    if np.isscalar(N):
        return g[shifted.columns].rolling(window=N, min_periods=1).sum() \
                .reset_index(level=0, drop=True)
    else:
        roll17 = g[shifted.columns].rolling(window=17, min_periods=1).sum() \
                    .reset_index(level=0, drop=True)
        roll18 = g[shifted.columns].rolling(window=18, min_periods=1).sum() \
                    .reset_index(level=0, drop=True)
        return roll17, roll18

def dynamic_window_all_attrs(df_grouped_weekly, attrs, mode='season_avg'):
    """
    Vectorized, multi-column version of your dynamic rolling logic.
    Expects df_grouped_weekly indexed by ['player_id','season','week']
    and containing the aggregated weekly columns `attrs`.
    Returns a DataFrame with SAME index and columns named per mode, e.g. 'season_avg_<attr>'
    """
    df = df_grouped_weekly.copy()
    shifted = _shift_group(df, attrs)

    out = None

    if mode == 'career_avg':
        # career expanding mean over shifted values
        g = shifted.groupby(level=0, sort=False)
        out = g[attrs].expanding(min_periods=1).mean().reset_index(level=0, drop=True)

    elif mode == 'form':
        # last-3 over career (shifted)
        g = shifted.groupby(level=0, sort=False)
        out = g[attrs].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    elif mode in ('season_avg','season_total'):
        # default within-season expanding on SHIFTED
        within = _within_season_expanding_sum_mean(
            shifted, how=('sum' if mode=='season_total' else 'mean')
        )

        # Week-1 override: career rolling last N = 17 before 2021, else 18 (exactly your first function)
        week = df.index.get_level_values(2).to_numpy()
        season = df.index.get_level_values(1).to_numpy()
        is_week1 = (week == 1)
        if is_week1.any():
            use18 = (season >= 2021)
            # compute both, then pick positions
            if mode == 'season_avg':
                roll17, roll18 = _career_rolling_lastN(shifted, N=[17,18])
            else:
                roll17, roll18 = _career_rolling_lastN_sum(shifted, N=[17,18])

            # start from within and replace by position
            out = within.copy()
            # positional indices for week1 rows
            pos = np.nonzero(is_week1)[0]
            # assign per column using .iloc to avoid boolean alignment issues
            out.iloc[pos] = np.where(use18[pos, None], roll18.iloc[pos], roll17.iloc[pos])
        else:
            out = within

    else:
        # default to 'form'
        g = shifted.groupby(level=0, sort=False)
        out = g[attrs].rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)

    # rename columns with mode prefix
    out = out.add_prefix(f'{mode}_')
    return out