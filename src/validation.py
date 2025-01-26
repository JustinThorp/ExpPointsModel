import numpy as np
import torch
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from model import EPModel, features

model = EPModel()
model.load_state_dict(torch.load("data/mode.ph", weights_only=True))
model.eval()
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
            pl.col("posteam_score"),
            pl.col('season'),
            pl.col("defteam_score"),
            pl.col("home_score"),
            pl.col("away_score"),
            pl.when(pl.col("home_team") == pl.col("posteam"))
            .then(pl.col("posteam_score"))
            .otherwise(pl.col("defteam_score"))
            .alias("total_home_score"),
            pl.when(pl.col("away_team") == pl.col("posteam"))
            .then(pl.col("posteam_score"))
            .otherwise(pl.col("defteam_score"))
            .alias("total_away_score"),
            pl.when(pl.col("home_team") == pl.col("posteam"))
            .then(1)
            .otherwise(0)
            .alias("homepos"),
            pl.col("play_type"),
            pl.col("home_team"),
            pl.col("away_team"),
            pl.col("epa"),
            pl.col("spread_line"),
            pl.col("total_line"),
        ]
    )
    .drop_nulls()
    .collect()
)

preds = (
    model(torch.tensor(data[features].to_numpy(), dtype=torch.float32))
    .squeeze()
    .detach()
    .numpy()
)

data = (
    data.with_columns_seq(
        pl.lit(preds[:, 0]).alias("HomeEP"),
        pl.lit(preds[:, 1]).alias("AwayEP"),
        pl.lit(preds[:, 0] - preds[:, 1]).alias("HomeEPD"),
        pl.lit(preds[:, 1] - preds[:, 0]).alias("AwayEPD"),
    )
    .with_columns(
        pl.col("HomeEP").shift(-1).alias("HomeEPA"),
        pl.col("AwayEP").shift(-1).alias("AwayEPA"),
        pl.col("HomeEPD").shift(-1).alias("HomeEPDA"),
        pl.col("AwayEPD").shift(-1).alias("AwayEPDA"),
    )
    .with_columns(
        pl.col("HomeEPA").sub(pl.col("HomeEP")).alias("HomeEPA"),
        pl.col("AwayEPA").sub(pl.col("AwayEP")).alias("AwayEPA"),
        pl.col("HomeEPDA").sub(pl.col("HomeEPD")).alias("HomeEPDA"),
        pl.col("AwayEPDA").sub(pl.col("AwayEPD")).alias("AwayEPDA"),
    )
    .with_columns(
        pl.when(pl.col("posteam") == pl.col("home_team"))
        .then(pl.col("HomeEPA"))
        .otherwise(pl.col("AwayEPA"))
        .alias("MyEPA"),
        pl.when(pl.col("posteam") == pl.col("home_team"))
        .then(pl.col("HomeEPDA"))
        .otherwise(pl.col("AwayEPDA"))
        .alias("MyEPDA"),
    )
)


output_df = (
    data.filter(pl.col("qtr") <= 4)
    .select(
        pl.col("qtr"),
        pl.col("HomeEP"),
        pl.col("home_score"),
        pl.col("HomeEP").sub(pl.col("home_score")).pow(2).alias("HomeSE"),
        pl.col("AwayEP").sub(pl.col("away_score")).pow(2).alias("AwaySE"),
    )
    .group_by("qtr")
    .agg(pl.col("HomeSE").mean(), pl.col("AwaySE").mean())
    .sort("qtr")
)
print(output_df)
