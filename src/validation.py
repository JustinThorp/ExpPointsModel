import numpy as np
import torch
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from model import EPModel, features

print(features)
model = EPModel()
model.load_state_dict(torch.load("data/mode.ph", weights_only=True))
model.eval()

data = (
    pl.select(
        [
            pl.lit(1).alias("down"),
            pl.lit(10).alias("ydstogo"),
            pl.lit(1).alias("qtr"),
            pl.lit(list(range(1, 100))).alias("yardline_100"),
            pl.lit([3600]).alias("game_seconds_remaining"),
            pl.lit(0).alias("total_pos_score"),
        ]
    )
    .explode("yardline_100")
    .explode("game_seconds_remaining")
)

preds = (
    model(torch.tensor(data[features].to_numpy(), dtype=torch.float32))
    .squeeze()
    .detach()
    .numpy()
)
data = data.with_columns(pl.lit(preds).alias("EP"))
print(data.to_pandas().info())
table = (
    data.group_by("game_seconds_remaining")
    .agg(pl.col("EP").min().alias("MinEP"), pl.col("EP").max().alias("MaxEP"))
    .with_columns(pl.col("MaxEP").sub(pl.col("MinEP")).alias("Diff"))
)
print(table)
fig, ax = plt.subplots()

sns.lineplot(x="yardline_100", y="EP", data=data, ax=ax, hue="game_seconds_remaining")
plt.show()
