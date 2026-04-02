# to calculate total SCORE
import pandas as pd
import numpy as np

# setting tree weights
TREE_WEIGHTS = {
    "cost": 0.35,
    "demand": 0.30,
    "accessibility": 0.25,
    "competition": 0.1
}

# round to 2 dp
def safe_float(v, decimals = 2):
    try:
        return round(float(v), decimals)
    except (TypeError, ValueError):
        return 0.0
    
# standardise/normalise the data to all of singapore
# add inversion for things that aren't necessarily better if it's higher (aka rent, etc.)
def minmax_series(series: pd.Series, invert = False) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mn == mx:
        return pd.Series(50.0, index = series.index)
    norm = (series - mn) / (mx - mn) * 100
    return (100 - norm) if invert else norm

# scoring for eahc subcategory
"""
returns a df with the four score columns and final_score
to a copy of the planning area df.
this happens after demand_score is calculated and the demand_Score
column has been added to the df.
scores are between 0-100 and higher is better
"""
def score_area_df(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # COST SCORE
    # lower rent is better, so invert!
    df["score_cost"] = minmax_series(df["avg_median_rent_psm"], invert= True)

    # DEMAND SCORE
    # alr got
    df["score_demand"] = df["demand_score"].clip(0,100)

    # ACCESSIBILITY SCORE
    # more mrt and bus is good so no invert!
    combined_accessibility = (
        df["avg_mrt_exits_500m"].fillna(0) + df["avg_bus_stops_500m"].fillna(0)
    )
    df["score_accessibility"] = minmax_series(combined_accessibility)

    # COMPETITIVENESS SCORE
    # fewer f&b outlets in the area is better, so invert!
    df["score_competition"] = minmax_series(df["total_establishments"], invert= True)

    # APPLY THE TREE WEIGHTS
    df["final_score"] = (
        df["score_cost"] * TREE_WEIGHTS["cost"]
        + df["score_demand"] * TREE_WEIGHTS["demand"]
        + df["score_accessibility"] * TREE_WEIGHTS["accessibility"]
        + df["score_competition"] * TREE_WEIGHTS["competition"]
    )

    return df

# decision tree filtering
"""
applies the four decision-tree gates in order and then returns a sorted df
node 1 - cost (HARD filter by rent budget)
node 2 - demand (SOFT filter, drop the bottom 25 pct of the remaining)
node 3 - accessibility (optional region HARD filter)
node 4 - competitiveness (score & rank by the filter)
"""
def apply_decision_tree(area_df: pd.DataFrame, min_rent: float, max_rent: float, region: str) -> pd.DataFrame:

    # NODE 1: COST
    filtered = area_df[
        (area_df["avg_median_rent_psm"] >= min_rent) & (area_df["avg_median_rent_psm"] <= max_rent)
    ].copy()

    if filtered.empty:
        return filtered

    # NODE 2: DEMAND
    demand_threshold = filtered["demand_score"].quantile(0.25)
    filtered = filtered[filtered["demand_score"] >= demand_threshold]

    # NODE 3: ACCESSIBILITY
    if region:
        loc_filtered = filtered[
            filtered["region"].str.strip().str.lower() == region.lower()
            ]
        if not loc_filtered.empty:
            filtered = loc_filtered
    return filtered

# format the data
def area_to_dict(row: pd.Series) -> dict:
    return{
        "planning_area": row["planning_area_name"],
        "region": row["region"].strip(),
        "avg_rent_psm": safe_float(row["avg_median_rent_psm"]),
        "num_competitors": int(row["total_establishments"]),
        "avg_mrt_exits": safe_float(row.get("avg_mrt_exits_500m", 0)),
        "avg_bus_stops": safe_float(row.get("avg_bus_stops_500m", 0)),
        "scores": {
            "cost": round(float(row["score_cost"]), 1),
            "demand": round(float(row["score_demand"]), 1),
            "accessibility": round(float(row["score_accessibility"]), 1),
            "competition": round(float(row["score_competition"]), 1),
            "overall": round(float(row["final_score"]), 1),
        },
        "demand_detail":{
            "income_score": safe_float(row.get("income_score", 0) * 100),
            "population_score": safe_float(row.get("population_score", 0) * 100),
            "working_age_score": safe_float(row.get("working_age_score", 0) * 100),
            "affluence_score": safe_float(row.get("affluence_score", 0) * 100),
        }
    }
