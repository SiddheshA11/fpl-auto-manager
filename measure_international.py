"""Do minutes change in the gameweek after an international break?

Breaks are not labelled anywhere, but they are visible in the fixture
calendar: a normal gameweek follows the last by about seven days, an
international break stretches that to a fortnight.

FPL's `region` is a bare numeric country id with no lookup shipped in the
dataset. It is an alphabetical country index, and REGION_* below were each
confirmed by reading off a player whose nationality is not in doubt, never
guessed from the ordering - two early probes matched the wrong player (region
21 resolves to Belgium via Amadou Onana, not Cote d'Ivoire; a search for
"Erling" matched "Sterling"), so every id here is one checked by name.

`region` is only present in 2024-25 onward, but nationality is static, so the
map is built on `code` - the stable cross-season player id the rest of the
codebase keys on - and applied backward through each season's id->code table.

CONTROL: ordinary gameweeks, same seasons, same players.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

import priors

H=Path(__file__).parent/"data"/"history"

# Confederation matters more than country: what costs minutes is the flight and
# the tournament, not the flag. CONMEBOL and CAF players cross an ocean; the
# home nations mostly play at Wembley.
CONMEBOL = {10: "Argentina", 30: "Brazil", 48: "Colombia", 62: "Ecuador",
            168: "Paraguay", 230: "Uruguay"}
CAF = {38: "Cameroon", 50: "DR Congo", 54: "Cote d'Ivoire", 63: "Egypt",
       81: "Ghana", 132: "Mali", 157: "Nigeria"}
FAR = {13: "Australia", 107: "Jamaica", 114: "South Korea", 229: "USA"}
HOME = {241: "England", 242: "N Ireland", 243: "Scotland", 244: "Wales",
        104: "Ireland"}


def confederation(region: float | None) -> str | None:
    """Travel burden band for an FPL region id."""
    if region is None or region != region:      # NaN
        return None
    r = int(region)
    if r in CONMEBOL:
        return "CONMEBOL"
    if r in CAF:
        return "CAF"
    if r in FAR:
        return "far AFC/CONCACAF"
    if r in HOME:
        return "home nations"
    return "UEFA other"
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
