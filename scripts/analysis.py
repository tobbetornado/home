# scripts/make_figure.py
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize

# Paths (can be overridden by env)
DATA_PATH = os.environ.get("DATA_PATH", "data/data.csv")
OUT_DIR = os.environ.get("OUT_DIR", "artifact")
OUT_FILE = os.path.join(OUT_DIR, "fermentering_figure.png")

os.makedirs(OUT_DIR, exist_ok=True)

def read_csv_flexible(path):
    # Try semicolon and comma separated variants and common encodings
    encs = ["utf-8", "latin1", "cp1252"]
    seps = [";", ",", "\t"]
    for enc in encs:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                # must have at least two columns to be useful
                if df.shape[1] >= 1:
                    return df
            except Exception:
                continue
    # last attempt with default pandas
    return pd.read_csv(path)

df = read_csv_flexible(DATA_PATH)

# Normalize column names (lower, strip)
df.columns = [c.strip() for c in df.columns]

# Heuristics to find columns
cols = [c.lower() for c in df.columns]
# datetime may be split into date + time
date_col = next((c for c in df.columns if c.lower().startswith("date")), None)
time_col = next((c for c in df.columns if c.lower().startswith("time")), None)
datetime_col = None
if date_col:
    if time_col:
        df["tid"] = pd.to_datetime(df[date_col].astype(str).str.strip() + "T" + df[time_col].astype(str).str.strip(), errors="coerce")
    else:
        df["tid"] = pd.to_datetime(df[date_col], errors="coerce")
else:
    # look for ISO-like columns
    for c in df.columns:
        s = df[c].astype(str)
        if s.str.contains("T").any() or s.str.contains(r"\d{4}-\d{2}-\d{2}").any():
            df["tid"] = pd.to_datetime(s, errors="coerce")
            break

# gravity column
gravity_col = next((c for c in df.columns if "g" in c.lower() or "gravity" in c.lower() or "g (målt)" in c.lower()), None)
# temperature
temp_col = next((c for c in df.columns if "temp" in c.lower() or "temper" in c.lower()), None)

# fallback positional columns
if "tid" not in df.columns:
    df["tid"] = pd.to_datetime(df.iloc[:,0], errors="coerce")

if gravity_col is None:
    # try to find numeric column with values ~0.9-1.2
    for c in df.columns:
        vals = pd.to_numeric(df[c].astype(str).str.replace(",", ".").str.replace(" ", ""), errors="coerce")
        if vals.notna().any() and vals.between(0.9, 1.2).any():
            gravity_col = c
            break

if temp_col is None:
    for c in df.columns:
        vals = pd.to_numeric(df[c].astype(str).str.replace(",", ".").str.replace(" ", ""), errors="coerce")
        if vals.notna().any() and vals.between(-10, 50).any() and c != gravity_col:
            temp_col = c
            break

# Build tidy df
tid = df["tid"]
gravity = pd.to_numeric(df[gravity_col].astype(str).str.replace(",", ".").str.replace(" ", ""), errors="coerce") if gravity_col else pd.Series(np.nan, index=df.index)
temp = pd.to_numeric(df[temp_col].astype(str).str.replace(",", ".").str.replace(" ", ""), errors="coerce") if temp_col else pd.Series(np.nan, index=df.index)

tid = pd.to_datetime(tid, errors="coerce")
tid_mask = ~tid.isna()
tid = tid[tid_mask]
gravity = gravity[tid_mask].reset_index(drop=True)
temp = temp[tid_mask].reset_index(drop=True)
tid = tid.reset_index(drop=True)

if len(tid) == 0:
    raise SystemExit("No valid timestamps found in the CSV. Check datetime columns.")

df2 = pd.DataFrame({"tid": tid, "gravity": gravity, "temp": temp})
df2 = df2.sort_values("tid").reset_index(drop=True)
df2 = df2.dropna(subset=["gravity"])  # remove rows without gravity (can't fit)
if df2.empty:
    raise SystemExit("No numeric gravity values found after cleaning. Check decimal separators (',' vs '.') and empty cells.")

df2["delta_days"] = (df2["tid"] - df2["tid"].iloc[0]).dt.total_seconds() / (24 * 3600)

OG = 1.065
FG = 1.015

def modell1(time, r, t_0):
    return FG + (OG - FG)/(1 + np.exp(r*(time - t_0)))
def modell2(time, r, t_0, s):
    return FG + (OG - FG)/((1 + s*np.exp(r*(time - t_0)))**(1.0/s))
def modell3(time, r, t_0):
    return ((OG-FG)/OG)/(1 + np.exp(r*(time - t_0)))


xdata = df2["delta_days"].values
ydata = df2["gravity"].values

# initial guess and bounds
p1 = [1.0, np.median(xdata)]
bounds = ([-10, -100], [10, 100])

popt1, pcov1 = scipy.optimize.curve_fit(modell1, xdata, ydata, p0=p1, bounds=bounds, maxfev=10000)
r_fit1, t0_fit1 = popt1
fitted = True

p2 = [1.0, np.median(xdata), 1.0]
bounds = ([-10, -100, -10], [10, 100, 10])

popt2, pcov2 = scipy.optimize.curve_fit(modell2, xdata, ydata, p0=p2, bounds=bounds, maxfev=10000)
r_fit2, t0_fit2, s_fit2 = popt2
fitted = True

p3 = [1.0, np.median(xdata)]
bounds = ([-10, -100], [10, 100])

popt3, pcov3 = scipy.optimize.curve_fit(modell3, xdata, ydata, p0=p3, bounds=bounds, maxfev=10000)
r_fit3, t0_fit3 = popt3
fitted = True

x_modell = np.linspace(0, max(20, xdata.max()+10), 200)
G_modell1 = modell1(x_modell, r_fit1, t0_fit1)
G_modell2 = modell2(x_modell, r_fit2, t0_fit2, s_fit2)
G_modell3 = modell3(x_modell, r_fit3, t0_fit3)



# Plot (matplotlib)
fig, ax1 = plt.subplots(figsize=(8,5))
ax1.scatter(xdata, ydata, label="Gravity measured")
ax1.plot(x_modell, G_modell1, label="Gravity model")
ax1.plot(x_modell, G_modell2, label="Gravity model med s")
ax1.plot(x_modell, G_modell3, label="Gravity model 3")

x_line = np.array([0, max(x_modell)])
og_line = np.array([OG, OG])
fg_line = np.array([FG, FG])

ax1.plot(x_line, og_line, linestyle="--", label="Original gravity")
ax1.plot(x_line, fg_line, linestyle="--", label="Predicted final gravity")


ax1.set_xlabel("Tid i dager")
ax1.set_ylabel("Gravity")

ax3 = ax1.twinx()
ax3.plot(xdata, df2["temp"].values, linestyle="--", label="Temperature")
ax3.set_ylabel("temperatur [°C]")

plt.title("Fermentering")
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
fig.legend(lines+lines2, labels+labels2, loc="upper right", bbox_to_anchor=(0.82, 0.9))

fig.savefig(OUT_FILE, bbox_inches="tight")
print("Saved:", OUT_FILE)
