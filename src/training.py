import polars as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from model import features, EPModel

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
            pl.col("season"),
            pl.col("yardline_100"),
            pl.col("posteam_score"),
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
            pl.col("spread_line"),
            pl.col("total_line"),
        ]
    )
    .drop_nulls()
    .collect()
)

print(data)


class TrainingData(Dataset):
    def __init__(self, features: pl.DataFrame, target: pl.DataFrame) -> None:
        super().__init__()
        self.features = torch.tensor(features.to_numpy(), dtype=torch.float32)
        self.target = torch.tensor(target.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.target[index]


train_dataset = TrainingData(
    data[features],
    data[["home_score", "away_score"]],
)
train_dataloader = DataLoader(train_dataset, batch_size=10000, shuffle=True)


net = EPModel()

criterion = nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=0.008, weight_decay=0.0)


for i in range(50):
    avg_loss = 0
    print(f"EPOCH {i}")
    print("------------")
    for i, (X, y) in enumerate(train_dataloader):
        pred = net(X)
        loss = criterion(y, pred)
        loss.backward()
        if i % 20 == 0:
            print(f"Loss: {loss.item():2f}")
        optim.step()
        optim.zero_grad()


torch.save(net.state_dict(), "data/mode.ph")
