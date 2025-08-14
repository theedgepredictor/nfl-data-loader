import re
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import datetime
import os
from typing import List, Optional
import pyarrow as pa
import requests
from bs4 import BeautifulSoup
from pandas.core.dtypes.common import is_numeric_dtype


def get_dataframe(path: str, columns: List = None):
    """
    Read a DataFrame from a parquet file.

    Args:
        path (str): Path to the parquet file.
        columns (List): List of columns to select (default is None).

    Returns:
        pd.DataFrame: Read DataFrame.
    """
    try:
        return pd.read_parquet(path, engine='pyarrow', dtype_backend='numpy_nullable', columns=columns)
    except Exception as e:
        print(e)
        return pd.DataFrame()


def put_dataframe(df: pd.DataFrame, path: str):
    """
    Write a DataFrame to a parquet file.

    Args:
        df (pd.DataFrame): DataFrame to write.
        path (str): Path to the parquet file.
        schema (dict): Schema dictionary.

    Returns:
        None
    """
    key, file_name = path.rsplit('/', 1)
    if file_name.split('.')[1] != 'parquet':
        raise Exception("Invalid Filetype for Storage (Supported: 'parquet')")
    os.makedirs(key, exist_ok=True)
    df.to_parquet(f"{key}/{file_name}",engine='pyarrow', schema=pa.Schema.from_pandas(df))


def create_dataframe(obj, schema: dict):
    """
    Create a DataFrame from an object with a specified schema.

    Args:
        obj: Object to convert to a DataFrame.
        schema (dict): Schema dictionary.

    Returns:
        pd.DataFrame: Created DataFrame.
    """
    df = pd.DataFrame(obj)
    for column, dtype in schema.items():
        df[column] = df[column].astype(dtype)
    return df

def get_seasons_to_update(root_path, feature_store_name):
    """
    Get a list of seasons to update based on the root path and sport.

    Args:
        root_path (str): Root path for the sport data.
        sport (ESPNSportTypes): Type of sport.

    Returns:
        List: List of seasons to update.
    """
    current_season = find_year_for_season()
    if os.path.exists(f'{root_path}/{feature_store_name}'):
        seasons = os.listdir(f'{root_path}/{feature_store_name}')
        fs_season = -1
        for season in seasons:
            temp = int(season.split('.')[0])
            if temp > fs_season:
                fs_season = temp
    else:
        fs_season = 2002
    if fs_season == -1:
        fs_season = 2002
    return list(range(fs_season, current_season + 1))


def find_year_for_season( date: datetime.datetime = None):
    """
    Find the year for a specific season based on the league and date.

    Args:
        league (ESPNSportTypes): Type of sport.
        date (datetime.datetime): Date for the sport (default is None).

    Returns:
        int: Year for the season.
    """
    SEASON_START_MONTH = {

        "NFL": {'start': 6, 'wrap': False},
    }
    if date is None:
        today = datetime.datetime.utcnow()
    else:
        today = date
    start = SEASON_START_MONTH["NFL"]['start']
    wrap = SEASON_START_MONTH["NFL"]['wrap']
    if wrap and start - 1 <= today.month <= 12:
        return today.year + 1
    elif not wrap and start == 1 and today.month == 12:
        return today.year + 1
    elif not wrap and not start - 1 <= today.month <= 12:
        return today.year - 1
    else:
        return today.year

def find_week_for_season(date: datetime.datetime = None):
    ESPN_NFL_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?dates={ymd}"

    @dataclass(frozen=True)
    class WeekResult:
        week: int
        season_year: int
        season_type_label: str  # "Preseason", "Regular Season", "Postseason", "Off Season"
        detail_label: str  # e.g., "Week 7", "Wild Card", "Super Bowl"
        source_url: str

    def _parse_iso_z(dt_str: str) -> datetime:
        # ESPN uses trailing 'Z' for UTC; datetime.fromisoformat expects +00:00
        if dt_str.endswith("Z"):
            dt_str = dt_str[:-1] + "+00:00"
        return datetime.datetime.fromisoformat(dt_str)

    def _noon_local(dt_local: datetime.date, tz: str) -> datetime:
        """Return local noon for the given date (to avoid ESPN's 07:00Z day boundary edge cases)."""
        return datetime.datetime(dt_local.year, dt_local.month, dt_local.day, 12, 0, 0, tzinfo=ZoneInfo(tz))

    def get_nfl_week_for_date(
            on_date: Optional[datetime.date | datetime.datetime] = None,
            *,
            tz: str = "America/New_York",
            timeout: int = 15
    ) -> WeekResult:
        """
        Determine the NFL 'week number' for a given date using ESPN's season calendar.

        Rules:
          - Preseason or Offseason -> 0
          - Regular Season -> that week's number (1..18 typically)
          - Postseason -> max_regular_week + (postseason_entry_index - 1)
            (e.g., Super Bowl -> 18 + 4 = 22, if regular season max is 18)

        Parameters
        ----------
        on_date : date | datetime | None
            If None, uses today's date in the provided timezone.
            If a datetime is naive, it's interpreted in the provided timezone.
        tz : str
            IANA timezone for interpreting dates (default America/New_York).
        timeout : int
            HTTP timeout in seconds.

        Returns
        -------
        WeekResult
            week: computed week number per rules
            season_year, season_type_label, detail_label, source_url
        """
        # Normalize to a local date
        if on_date is None:
            local_dt = datetime.datetime.now(ZoneInfo(tz))
        elif isinstance(on_date, datetime.datetime):
            local_dt = on_date if on_date.tzinfo else on_date.replace(tzinfo=ZoneInfo(tz))
            local_dt = local_dt.astimezone(ZoneInfo(tz))
        elif isinstance(on_date, datetime.date):
            local_dt = datetime.datetime(on_date.year, on_date.month, on_date.day, tzinfo=ZoneInfo(tz))
        else:
            raise TypeError("on_date must be date, datetime, or None")

        ymd = local_dt.strftime("%Y%m%d")
        url = ESPN_NFL_SCOREBOARD.format(ymd=ymd)

        # Fetch JSON
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()

        leagues = payload.get("leagues") or []
        if not leagues:
            raise ValueError("No 'leagues' data in ESPN response.")
        league = leagues[0]

        # Grab the season year for reference
        season_year = int(league.get("season", {}).get("year", local_dt.year))

        calendar_blocks = league.get("calendar") or []
        if not calendar_blocks:
            raise ValueError("No 'calendar' data in ESPN response.")

        # Compute max regular season week
        max_regular_week = 0
        for block in calendar_blocks:
            label = (block.get("label") or "").lower()
            if "regular" in label:
                entries = block.get("entries") or []
                for e in entries:
                    try:
                        max_regular_week = max(max_regular_week, int(e.get("value")))
                    except (TypeError, ValueError):
                        pass

        # Use local NOON to avoid ESPN's 07:00Z boundary edge cases
        local_noon = _noon_local(local_dt.date(), tz)
        noon_utc = local_noon.astimezone(datetime.timezone.utc)

        found = None  # (season_type_label, entry)
        # First try to match against entry windows
        for block in calendar_blocks:
            season_type_label = block.get("label") or ""
            for entry in (block.get("entries") or []):
                try:
                    start = _parse_iso_z(entry["startDate"])
                    end = _parse_iso_z(entry["endDate"])
                except KeyError:
                    # If entry is missing dates, fall back to block-level dates below
                    continue
                if start <= noon_utc < end:
                    found = (season_type_label, entry)
                    break
            if found:
                break

        # Fallback: use block-level window if no entry matched
        if not found:
            for block in calendar_blocks:
                season_type_label = block.get("label") or ""
                start = block.get("startDate")
                end = block.get("endDate")
                if start and end:
                    start = _parse_iso_z(start)
                    end = _parse_iso_z(end)
                    if start <= noon_utc < end:
                        # fabricate a minimal entry
                        found = (season_type_label, {"value": None, "label": season_type_label})
                        break

        if not found:
            raise ValueError("Could not locate matching calendar entry for the given date.")

        season_type_label, entry = found
        st_lower = season_type_label.lower()
        detail_label = entry.get("label") or entry.get("alternateLabel") or season_type_label
        raw_value = entry.get("value")

        # Apply your week rules
        if "preseason" in st_lower or "off" in st_lower:
            week = 0
        elif "regular" in st_lower:
            if raw_value is None:
                raise ValueError("Regular Season entry missing 'value'.")
            week = int(raw_value)
        elif "post" in st_lower:
            if raw_value is None:
                raise ValueError("Postseason entry missing 'value'.")
            round_index = int(raw_value) - 1  # Wild Card -> 0, ..., Super Bowl -> 4
            if max_regular_week <= 0:
                # Sensible default if max wasn't discoverable
                max_regular_week = 18
            week = max_regular_week + round_index
        else:
            # Unknown season type—treat safely as 0
            week = 0

        return WeekResult(
            week=week,
            season_year=season_year,
            season_type_label=season_type_label,
            detail_label=detail_label,
            source_url=url,
        )
    week_result = get_nfl_week_for_date(date)
    week = week_result.week
    if week == 0:
        week = 1
    return week

def get_webpage_soup(html, html_attr=None, attr_key_val=None):
    if html:
        soup = BeautifulSoup(html, 'html.parser')
        if html_attr:
            soup = soup.find(html_attr, attr_key_val)
        return soup
    return None

def clean_string(s):
    if isinstance(s, str):
        return re.sub("[\W_]+",'',s)
    else:
        return s

def re_alphanumspace(s):
    if isinstance(s, str):
        return re.sub("^[a-zA-Z0-9 ]*$",'',s)
    else:
        return s

def re_braces(s):
    if isinstance(s, str):
        return re.sub("[\(\[].*?[\)\]]", "", s)
    else:
        return s

def re_numbers(s):
    if isinstance(s, str):
        n = ''.join(re.findall(r'\d+', s))
        return int(n) if n != '' else n
    else:
        return s


def name_filter(s):
    s = clean_string(s)
    s = re_braces(s)
    s = str(s)
    s = s.replace(' ', '').lower()
    return s