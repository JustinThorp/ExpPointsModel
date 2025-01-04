from torch import nn

features = [
    "down",
    "ydstogo",
    "qtr",
    "yardline_100",
    "game_seconds_remaining",
    "posteam_score",
    "defteam_score",
    "total_home_score",
    "total_away_score",
    "homepos",
    "season"
    # "spread_line",
    # "total_line",
]


class EPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(len(features)),
            nn.Linear(len(features), 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, 2),
        )

    def forward(self, x):
        x = self.layers(x)
        return x
