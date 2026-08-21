"""What actually happens to a player's minutes when he comes back?

Production assumes RAMP_UP = [0.55, 0.80]: 55% of normal start probability in
the first gameweek back, 80% in the second, full thereafter. Hand-set, never
measured, and it only fires when live news exists - so the backtest never
exercises it either.

An absence is fully observable without any availability flag: a run of
consecutive zero-minute gameweeks for someone who was an established starter
before it. That conflates injury with being dropped, which is why the result is
reported split by absence length - a one-week gap is mostly rotation, a
six-week gap is not.

CONTROL: established starters matched on the same gameweeks who did NOT have an
absence, so ordinary mid-season drift cannot be read as a ramp.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

import priors

H=Path(__file__).parent/"data"/"history"
SEASONS=["2020-21","2021-22","2022-23","2023-24","2024-25","2025-26"]
BASE_GW=5          # gameweeks of evidence needed to call someone a starter
BASELINE=60.0      # min/GW before the absence to qualify

rows=[]
for s in SEASONS:
    d=priors.read_season_csv(H/s/"merged_gw.csv")
    if "position" in d.columns: d=d[d["position"]!="AM"]
    g=d.groupby(["element","GW"],as_index=False).agg(minutes=("minutes","sum"))
    g["season"]=s; rows.append(g)
g=pd.concat(rows,ignore_index=True).sort_values(["season","element","GW"])
key=["season","element"]

records=[]; control=[]
for (s,pid),pg in g.groupby(key):
    pg=pg.sort_values("GW").reset_index(drop=True)
    mins=pg["minutes"].to_numpy(); gws=pg["GW"].to_numpy()
    i=BASE_GW
    while i < len(pg):
        base=mins[max(0,i-BASE_GW):i]
        if len(base)<BASE_GW or base.mean()<BASELINE:
            i+=1; continue
        if mins[i]==0:
            j=i
            while j<len(mins) and mins[j]==0: j+=1
            L=j-i                                  # absence length in gameweeks
            b=float(base.mean())
            for k in range(0,4):                   # gameweeks after return
                if j+k<len(mins):
                    records.append({"L":L,"back":k+1,"ratio":mins[j+k]/b,"base":b})
            i=j+1
        else:
            # control: a non-absent starter, same relative window
            for k in range(0,4):
                if i+k<len(mins):
                    control.append({"back":k+1,"ratio":mins[i+k]/float(base.mean())})
            i+=1

r=pd.DataFrame(records); c=pd.DataFrame(control)
print(f"{len(r):,} post-return observations, {len(c):,} control observations, "
      f"{len(SEASONS)} seasons\n")

r["band"]=pd.cut(r["L"],[0,1,2,3,5,8,50],
                 labels=["1 GW","2 GW","3 GW","4-5 GW","6-8 GW","9+ GW"])
print("Minutes on return, as a fraction of the player's own pre-absence average")
print(f"{'absence':<10}{'n':>7}" + "".join(f"{'+'+str(k):>9}" for k in range(1,5)))
print("-"*54)
for b,bg in r.groupby("band",observed=True):
    line=f"{str(b):<10}{len(bg)//4:>7}"
    for k in range(1,5):
        v=bg[bg["back"]==k]["ratio"]
        line+=f"{v.mean():>9.3f}" if len(v) else f"{'-':>9}"
    print(line)
line=f"{'CONTROL':<10}{len(c)//4:>7}"
for k in range(1,5):
    line+=f"{c[c['back']==k]['ratio'].mean():>9.3f}"
print(line)

print(f"\nProduction assumes {0.55:.2f} then {0.80:.2f} then full, for every absence length.")
print("\nRatio to control (what the ramp multiplier should actually be):")
print(f"{'absence':<10}" + "".join(f"{'+'+str(k):>9}" for k in range(1,5)))
print("-"*47)
for b,bg in r.groupby("band",observed=True):
    line=f"{str(b):<10}"
    for k in range(1,5):
        v=bg[bg["back"]==k]["ratio"]; cv=c[c["back"]==k]["ratio"].mean()
        line+=f"{v.mean()/cv:>9.3f}" if len(v) else f"{'-':>9}"
    print(line)
