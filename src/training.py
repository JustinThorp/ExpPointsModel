import polars as pl
import torch

raw_data = pl.scan_parquet(
    "/Users/justinthorp/Documents/NFLWinProb/data/traing_data.parquet"
)
data = (
    raw_data.select(
        [
            pl.col("game_id"),
            pl.col("posteam"),
            pl.col("defteam"),
            pl.col("game_seconds_remaining"),
            pl.col("qtr"),
            pl.col("down"),
            pl.col("ydstogo"),
            pl.col("yardline_100"),
            pl.when(pl.col("posteam") == pl.col("home_team"))
            .then(pl.col("home_score"))
            .otherwise(pl.col("away_score"))
            .alias("pos_score"),
            pl.when(pl.col("defteam") == pl.col("away_team"))
            .then(pl.col("away_score"))
            .otherwise(pl.col("home_score"))
            .alias("def_score"),
            pl.when(pl.col("posteam") == pl.col("home_team"))
            .then(pl.col("total_home_score"))
            .otherwise(pl.col("total_away_score"))
            .alias("total_pos_score"),
            pl.when(pl.col("defteam") == pl.col("away_team"))
            .then(pl.col("total_away_score"))
            .otherwise(pl.col("total_home_score"))
            .alias("total_def_score"),
        ]
    )
    .with_columns([])
    .collect()
)

print(data)
