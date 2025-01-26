import polars as pl
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import features, EPModel

model = EPModel()
model.load_state_dict(torch.load("data/mode.ph", weights_only=True))
model.eval()
raw_data = pl.scan_parquet(
    "/Users/justinthorp/Documents/NFLWinProb/data/traing_data.parquet"
)

GAMEID = "2024_01_DAL_CLE"


#print([x for x in raw_data.columns if "drive" in x])
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
            pl.col('season'),
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
            pl.col("play_type"),
            pl.col("home_team"),
            pl.col("away_team"),
            pl.col("epa"),
            pl.col("spread_line"),
            pl.col("total_line"),
        ]
    )
    .drop_nulls()
    .filter(pl.col("game_id") == GAMEID)
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
        pl.col("game_seconds_remaining").sub(3600).mul(-1),
    )
)


def style(fig, ax):
    # Set the background color
    fig.patch.set_facecolor('#f0f0f0')
    ax.set_facecolor('#f0f0f0')

    # Remove the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Color and style the remaining spines
    ax.spines['bottom'].set_color('#ccc8c8')
    ax.spines['left'].set_color('#ccc8c8')

    # Style the grid
    ax.grid(True, 'major', color='#dbdbdb', linestyle='-', linewidth=1)
    ax.grid(True, 'minor', color='#dbdbdb', linestyle='-', linewidth=0.5)

    # Put grid behind plots
    ax.set_axisbelow(True)

    # Style the ticks
    ax.tick_params(axis='both', colors='#666666', labelsize=10)

    # Set the default font
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']

    # Style the title and labels
    ax.set_title(ax.get_title(), color='#666666', size=14, pad=15)
    ax.set_xlabel(ax.get_xlabel(), color='#666666', size=10)
    ax.set_ylabel(ax.get_ylabel(), color='#666666', size=10)

    # Add padding to the layout
    fig.tight_layout(pad=2.0)

fig, ax = plt.subplots(1)
ax.plot(data["game_seconds_remaining"], data["HomeEP"])
ax.plot(data["game_seconds_remaining"], data["AwayEP"])
ax.fill_between(
    data["game_seconds_remaining"], data["HomeEP"], data["AwayEP"], alpha=0.2
)
ax.set_xlabel('Game Seconds Remaining')
ax.set_ylabel('Predicted Score')
ax.set_title('2024 Week 1 DAL @ CLE')
ax.axhline(33,linestyle = 'dashed',color = 'orange',alpha = .5)
ax.axhline(17,linestyle = 'dashed',color = 'blue',alpha = .5)
fig.patch.set_facecolor('#f0f0f0')
ax.set_facecolor('#f0f0f0')

# Remove the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Color and style the remaining spines
ax.spines['bottom'].set_color('#ccc8c8')
ax.spines['left'].set_color('#ccc8c8')

# Style the grid
ax.grid(True, 'major', color='#dbdbdb', linestyle='-', linewidth=1)
ax.grid(True, 'minor', color='#dbdbdb', linestyle='-', linewidth=0.5)

# Put grid behind plots
ax.set_axisbelow(True)

# Style the ticks
ax.tick_params(axis='both', colors='#666666', labelsize=10)

# Set the default font
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Style the title and labels
ax.set_title(ax.get_title(), color='#666666', size=14, pad=15)
ax.set_xlabel(ax.get_xlabel(), color='#666666', size=10)
ax.set_ylabel(ax.get_ylabel(), color='#666666', size=10)

# Add padding to the layout
fig.tight_layout(pad=2.0)
fig.patch.set_facecolor('#f0f0f0')
fig.savefig('data/game_plot.jpg',
                dpi=300,
                bbox_inches='tight',
                facecolor=fig.get_facecolor(),
                edgecolor='none')
