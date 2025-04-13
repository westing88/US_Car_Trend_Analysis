import pandas as pd
import numpy as np
import itertools
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings("ignore")

# === CONFIG ===
input_file = "/Users/weishiting/Desktop/243 Analytics Lab/Module2/matched_dataset.csv"  # <-- Your full dataset
output_file = "forecast_results.csv"
min_years_required = 1

# === LOAD & PREPROCESS ===
df = pd.read_csv(input_file)
make_cols = ["MAKE1", "MAKE2", "MAKE3", "MAKE4"]
year_cols = ["YEAR1", "YEAR2", "YEAR3", "YEAR4"]
df = df[["PID", "STATE"] + make_cols + year_cols]

# Melt long format
df_melted = df.melt(id_vars=["PID", "STATE"], value_vars=make_cols, var_name="MAKE_COL", value_name="MAKE")
df_years = df.melt(id_vars=["PID", "STATE"], value_vars=year_cols, var_name="YEAR_COL", value_name="YEAR")

df_melted["YEAR"] = df_years["YEAR"]
df_melted.dropna(subset=["MAKE", "YEAR"], inplace=True)
df_melted["YEAR"] = df_melted["YEAR"].astype(int)

# === FIND TOP 10 STATES BY TOTAL PURCHASES ===
top_states = (
    df_melted.groupby("STATE")
    .size()
    .sort_values(ascending=False)
    .head(10)
    .index.tolist()
)

forecast_records = []

# === PARAMETER GRID ===
p = d = q = range(0, 2)
P = D = Q = range(0, 2)
S = [3, 4, 12]
param_grid = list(itertools.product(p, d, q, P, D, Q, S))

# === LOOP THROUGH EACH TOP STATE & ITS TOP 3 BRANDS ===
for state in top_states:
    print(f"\n🔍 Processing state: {state}")

    state_df = df_melted[df_melted["STATE"] == state]
    top_brands = (
        state_df.groupby("MAKE")
        .size()
        .sort_values(ascending=False)
        .head(3)
        .index.tolist()
    )

    for brand in top_brands:
        print(f"  → Modeling {brand}")
        series_df = state_df[state_df["MAKE"] == brand]

        # Build the series

        ts = series_df.groupby("YEAR").size()
        ts = ts.sort_index()

        # Safely convert to datetime with year-end format
        ts.index = pd.date_range(start=f"{ts.index.min()}-12-31", periods=len(ts), freq="Y")

        # Fill in any missing years
        ts = ts.asfreq("Y", fill_value=0)

        print(f"✅ Verified series:\n{ts.tail(10)}")


        n = len(ts)
        
        train_size = int(n * 0.8)
        train_ts = ts[:train_size]
        val_ts = ts[train_size:]

        best_mse = float("inf")
        best_params = None

        for params in param_grid:
            try:
                model = SARIMAX(train_ts,
                                order=(params[0], params[1], params[2]),
                                seasonal_order=(params[3], params[4], params[5], params[6]),
                                enforce_stationarity=False,
                                enforce_invertibility=False).fit(disp=False)
                val_pred = model.forecast(steps=len(val_ts))
                mse = mean_squared_error(val_ts, val_pred)
                if mse < best_mse:
                    best_mse = mse
                    best_params = params
            except:
                continue
        
        # Final model & forecast
        if best_params:
            model = SARIMAX(ts,
                            order=(best_params[0], best_params[1], best_params[2]),
                            seasonal_order=(best_params[3], best_params[4], best_params[5], best_params[6]),
                            enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False)
            forecast_val = model.forecast(steps=1).iloc[0]

            # Save historical
            for year, value in ts.items():
                forecast_records.append({
                    "STATE": state,
                    "BRAND": brand,
                    "YEAR": year.year,
                    "PURCHASES": value,
                    "PREDICTED": None
                })

            # Save prediction
            forecast_records.append({
                "STATE": state,
                "BRAND": brand,
                "YEAR": ts.index[-1].year + 1,
                "PURCHASES": None,
                "PREDICTED": forecast_val
            })
            print(f"    Forecasted next year = {forecast_val:.1f}")


# === SAVE RESULTS ===
forecast_df = pd.DataFrame(forecast_records)
forecast_df.to_csv(output_file, index=False)
print(f"\n✅ Forecasts saved to: {output_file}")
