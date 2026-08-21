"""Do minutes change in the gameweek after an international break?

Breaks are not labelled anywhere, but they are visible in the fixture
calendar: a normal gameweek follows the last by about seven days, an
international break stretches that to a fortnight.

Nationality would be the interesting split - a player flying to South America
loses far more than one playing at Wembley - but FPL's `region` is a bare
numeric country id with no lookup shipped in the dataset, so that split would
be guesswork and is not attempted here.

CONTROL: ordinary gameweeks, same seasons, same players.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

import priors

H=Path(__file__).parent/"data"/"history"
SEASONS=["2021-22","2022-23","2023-24","2024-25","2025-26"]
out=[]
for s in SEASONS:
    d=priors.read_season_csv(H/s/"merged_gw.csv")
    if "position" in d.columns: d=d[d["position"]!="AM"]
    d["kick"]=pd.to_datetime(d["kickoff_time"],errors="coerce",utc=True)
    gwdate=d.groupby("GW")["kick"].median().sort_index()
    gap=gwdate.diff().dt.days
    after_break=set(gap[gap>=12].index.tolist())
    g=d.groupby(["element","GW"],as_index=False).agg(minutes=("minutes","sum"))
    g["season"]=s; g["after_break"]=g["GW"].isin(after_break)
    out.append(g)
    print(f"  {s}: gameweeks after a break -> {sorted(after_break)}")
g=pd.concat(out,ignore_index=True).sort_values(["season","element","GW"])
key=["season","element"]

# baseline: the player's own trailing 5-gameweek average, so this is a
# within-player comparison rather than a comparison of different players
g["base"]=g.groupby(key)["minutes"].transform(lambda x: x.shift(1).rolling(5).mean())
g=g[(g["base"]>=60)]                      # established starters only
g["ratio"]=g["minutes"]/g["base"]

a=g[g["after_break"]]; c=g[~g["after_break"]]
print(f"\n{'':<26}{'n':>9}{'mean minutes':>15}{'ratio to own base':>20}{'P(blank)':>11}")
print("-"*82)
for lab,f in [("gameweek after a break",a),("CONTROL: ordinary GW",c)]:
    print(f"{lab:<26}{len(f):>9}{f['minutes'].mean():>15.1f}{f['ratio'].mean():>20.3f}{(f['minutes']==0).mean():>11.3f}")
d=a["ratio"].mean()-c["ratio"].mean()
se=np.sqrt(a["ratio"].var()/len(a)+c["ratio"].var()/len(c))
print(f"\ndifference {d:+.4f}  SE {se:.4f}  t = {d/se:+.2f}")
print(f"in minutes, that is {(a['minutes'].mean()-c['minutes'].mean()):+.2f} per starter per gameweek")
