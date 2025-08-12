
# nfl-data-loader

**nfl-data-loader** is a robust Python package for accessing, merging, and transforming NFL sports data from multiple sources in an object-oriented, pipeline-friendly way. It provides a unified API for teams, players, events, and venues, with rich schema documentation and ready-to-use data loaders, transformers, and formatters.

### TheEdgePredictor Services



Upstream
- espn-api-orm (PyPi Package)
    - [![Team Data trigger](https://github.com/theedgepredictor/team-data-pump/actions/workflows/team_data_trigger.yaml/badge.svg)](https://github.com/theedgepredictor/team-data-pump/actions/workflows/team_data_trigger.yaml)
    - [![Venue Data trigger](https://github.com/theedgepredictor/venue-data-pump/actions/workflows/venue_data_trigger.yaml/badge.svg)](https://github.com/theedgepredictor/venue-data-pump/actions/workflows/venue_data_trigger.yaml)
    - [![Event Data trigger](https://github.com/theedgepredictor/event-data-pump/actions/workflows/event_data_trigger.yaml/badge.svg)](https://github.com/theedgepredictor/event-data-pump/actions/workflows/event_data_trigger.yaml)
        
    - odds-data-pump (TBD) 
    - [![Fantasy Data trigger](https://github.com/theedgepredictor/fantasy-data-pump/actions/workflows/fantasy_data_trigger.yaml/badge.svg)](https://github.com/theedgepredictor/fantasy-data-pump/actions/workflows/fantasy_data_trigger.yaml)
- nfl-madden-data (Manual Yearly Trigger)
- [![ELO-Rating Data trigger](https://github.com/theedgepredictor/elo-rating/actions/workflows/elo_data_trigger.yaml/badge.svg)](https://github.com/theedgepredictor/elo-rating/actions/workflows/elo_data_trigger.yaml)

Downstream
- [![Feature Store Data trigger](https://github.com/theedgepredictor/nfl-feature-store/actions/workflows/feature_store_data_trigger.yaml/badge.svg)](https://github.com/theedgepredictor/nfl-feature-store/actions/workflows/feature_store_data_trigger.yaml)
- nfl-model-store (TBD)
- nfl-madden-data (Manual Yearly Trigger)


## 🚀 Features
- Unified access to NFL data from multiple sources (nflverse, Madden, Pro Football Reference, ESPN API)
- Modular loader, transform, and formatter architecture
- Pydantic schemas for all endpoints and data objects
- Grouped by: `team`, `player`, `event`, `venue`
- Easy integration into ML/data pipelines
- Auto-deploy to PyPI on merge to main (see CI/CD)

## 📦 Data Sources
- [feature-store](https://github.com/theedgepredictor/nfl-feature-store)
- [Madden EA / madden.weebly](https://github.com/theedgepredictor/nfl-madden-data)
- [Pro Football Reference](https://www.pro-football-reference.com/)
- [ESPN API](https://github.com/theedgepredictor/espn-api-orm), [fantasy-data-pump](https://github.com/theedgepredictor/fantasy-data-pump)
- [nflverse](https://github.com/nflverse/nflverse-data)
- [greerreNFL](https://github.com/greerreNFL)

## 🏗️ Architecture
```
┌────────────┐   ┌────────────┐   ┌────────────┐
│  Sources   │→→ │ Formatters │→→ │  Workflows │
└────────────┘   └────────────┘   └────────────┘
      │               │                │
   [team, player, event, venue]  [Extract, Transform, Load Blocks and Pipelines]
```

## 🧩 Loader Groups
- **Team**: Team stats, rosters, schedules, advanced metrics
- **Player**: Player stats, depth charts, Madden ratings, combine data
- **Event**: Game-level stats, play-by-play, Vegas lines, EPA, fantasy
- **Venue**: Stadium/venue metadata and mappings

## 📑 Schemas & Data Dictionary
All endpoints return data validated by Pydantic schemas. See `/schemas` for:
- Field names, types, and descriptions (see `schemas/events/features.py`, `schemas/players/madden.py`, `schemas/players/position.py`)
- Example: Player Madden schema, event feature schema, position mappers

## 🛠️ Usage Example
```python
from nfl_data_loader.api.sources.players.rosters.rosters import collect_roster
from nfl_data_loader.workflows.transforms.players.player import make_player_stats

# Load player roster for a season
roster_df = collect_roster(2024)

# Transform player stats
player_stats = make_player_stats(2024, week=1, position_group='quarterback')
```

## 📈 Data Flow Example
1. **Load**: Pull raw data from any source (e.g., ESPN, nflverse) or pull cached feature store (Event, Player)
2. **Transform**: Apply rolling averages, imputation, feature engineering
3. **Format**: Output as DataFrame, JSON, or for ML pipelines

## 🧪 Testing & Validation
- All endpoints are covered by tests (see `/tests`)
- Schemas are validated with Pydantic
- Linting and type checking enabled

## 🤖 CI/CD & PyPI Deployment
- GitHub Actions workflow auto-deploys to PyPI on merge to `main`
- See `.github/workflows/publish.yml` for details

## 🤝 Contributing
Pull requests welcome! Please add/expand tests and update schema docs for new endpoints.

---

For a full list of available endpoints, schemas, and data fields, see the `/schemas` and `/api` directories. For questions or issues, open a GitHub issue.

## Publishing

Auto publish available through GitHub Actions and Pypi

Local publishing (include token in rc)

1. ```python -m build```

2. ```twine upload dist/*```