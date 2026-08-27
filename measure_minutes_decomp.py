"""Minutes decomposition, this time against the model production runs."""
import sys, logging
from pathlib import Path
import numpy as np, pandas as pd

import priors, xp_model as X, backtest as B
logging.basicConfig(level=logging.WARNING)

def collect(season, start, end, with_lags=True):
    sd=B.HISTORY_DIR/season
    gw=priors.read_season_csv(sd/"merged_gw.csv"); raw=priors.read_season_csv(sd/"players_raw.csv")
    tm=priors.read_season_csv(sd/"teams.csv"); fx=priors.read_season_csv(sd/"fixtures.csv").to_dict("records")
    cfg=B._scoring_config()
    earlier=[s for s in priors.available_seasons() if s<season]
    ps=priors.build_priors(seasons=earlier,current_team_codes={int(r["code"]):r["name"] for _,r in tm.iterrows()})
    out=[]
    for g in range(start,end+1):
        rows=gw[gw["GW"]==g]
        if rows.empty: continue
        act=rows.groupby("element",as_index=False).agg(minutes=("minutes","sum"),total_points=("total_points","sum"))
        st=B.build_state(gw,raw,tm,g,cfg)
        kw={"recent_minutes":B.recent_minutes_for(gw,g)} if with_lags else {}
        m=X.XPModel(st,fx,ps,X.ModelConfig(horizon=1),**kw)
        pred=m.expected_points([g])[["id","exp_minutes","p_start","cost","position"]]
        mm=act.merge(pred,left_on="element",right_on="id",how="inner"); mm["gw"]=g
        past=gw[gw["GW"]<g]
        if len(past):
            pm=past.groupby("element")["minutes"].agg(["sum","count"])
            mm["sd_min_per_gw"]=mm["element"].map(pm["sum"]/pm["count"]).fillna(0.0)
        else: mm["sd_min_per_gw"]=0.0
        out.append(mm)
    return pd.concat(out,ignore_index=True)

def r2(p,a):
    ss=float(((p-a)**2).sum()); st=float(((a-a.mean())**2).sum())
    return 1-ss/st if st>0 else float("nan")

for label,lags in [("WITHOUT lags (what I measured before)",False),("WITH lags (production model)",True)]:
    df=collect("2025-26",10,38,with_lags=lags)
    print(f"\n{'='*62}\n{label}\nn={len(df):,}  minutes R2 = {r2(df['exp_minutes'],df['minutes']):.4f}")
    df["role"]=pd.cut(df["sd_min_per_gw"],[-.1,5,30,60,80,91],
        labels=["never plays","fringe <30","rotation 30-60","regular 60-80","ever-present 80+"])
    tot=float(((df["exp_minutes"]-df["minutes"])**2).sum())
    print(f"{'segment':<20}{'n':>7}{'share err':>11}{'pred':>8}{'actual':>8}{'bias':>8}")
    print("-"*62)
    for nme,g2 in df.groupby("role",observed=True):
        e=float(((g2["exp_minutes"]-g2["minutes"])**2).sum())
        print(f"{str(nme):<20}{len(g2):>7}{e/tot:>11.3f}{g2['exp_minutes'].mean():>8.1f}"
              f"{g2['minutes'].mean():>8.1f}{g2['exp_minutes'].mean()-g2['minutes'].mean():>8.1f}")
    reg=df[df["sd_min_per_gw"]>=60]; zero=reg[reg["minutes"]==0]
    print(f"regular blanks: {len(zero)}/{len(reg)} rows, "
          f"{float(((zero['exp_minutes']-zero['minutes'])**2).sum())/tot:.3f} of all squared error, "
          f"predicted at {zero['exp_minutes'].mean():.1f} min")
