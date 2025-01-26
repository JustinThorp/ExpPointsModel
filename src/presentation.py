import polars as pl



data = pl.read_parquet('data/scored_data.parquet')


print(data)
