import polars as pl
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

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
    data[["down", "yardline_100", "game_seconds_remaining", "total_pos_score"]],
    data[["pos_score"]],
)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)


class EPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


net = EPModel()

criterion = nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=0.1)


for i in range(10):
    avg_loss = 0
    i = 0
    for X, y in train_dataloader:
        pred = net(X)
        loss = criterion(y, pred)
        loss.backward()
        avg_loss += loss.item()
        i += 1
    optim.step()
    optim.zero_grad()
    print(avg_loss / i)
