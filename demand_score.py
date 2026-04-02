# to calculate the DEMAND score
import pandas as pd
import numpy as np

# setting weights for each sub-component
WEIGHTS = {
    "income": 0.35,
    "population": 0.25,
    "working_age": 0.25,
    "affluence": 0.15
}

# income brackets above median for spending power
HIGH_INCOME_BRACKETS = {
    "$11,000 - $11,999", "$12,000 - $12,999", "$13,000 - $13,999",
    "$14,000 - $14,999", "$15,000 - $17,499", "$17,500 - $19,999",
    "$20,000 & Over"
}

# age groups in the working-age, 20-64
WORKING_AGE_GROUPS = {
    "20-24", "25-29", "30-34", "35-39", "40-44",
    "45-49", "50-54", "55-59", "60-64",
}

# private houses
AFFLUENT_DWELLINGS = {"Condominiums and Apartments", "Landed Properties"}

# standardizes the score to every planning area score
def _minmax(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mn == mx:
        return pd.Series(0.5, index = series.index)
    return (series - mn)/(mx - mn)

# the mother code lol
"""
returns a df with the cols:
planning_area, demand_score, income_score, population_score
working_age_score, affluence_score
with ONE row per planning area
"""
def build_demand_scores(data_dir: str = "data/demographics") -> pd.DataFrame:

    # INCOME SCORE
    # preprocessing data
    income = pd.read_excel(f"{data_dir}/income.xlsx")
    income["area"] = income["Planning Area of Residence"].str.strip().str.lower()
    # segment the high earners
    income["is_high"] = income["Monthly Household Income"].isin(HIGH_INCOME_BRACKETS)
    # group by planning area and find the ratio to everybody
    income_grouped = income.groupby("area").apply(
        lambda g: g.loc[g["is_high"], "Number of Households"].sum() / g["Number of Households"].sum()
        if g["Number of Households"].sum() > 0 else 0
    ).reset_index()
    income_grouped.columns = ["area", "income_pct"]

    # POPULATION SCORE
    # preprocessing data
    pop = pd.read_excel(f"{data_dir}/population_ethnic.xlsx")
    pop["area"] = pop["Planning Area of Residence"].str.strip().str.lower()
    # sum by planning area
    pop_total = pop.groupby("area")["Number of Residents"].sum().reset_index()
    pop_total.columns = ["area", "total_pop"]

    # WORKING AGE SCORE
    # preprocessing data
    age = pd.read_excel(f"{data_dir}/population_age.xlsx")
    age["area"] = age["Planning Area of Residence"].str.strip().str.lower()
    age["is_working"] = age["Age Group"].isin(WORKING_AGE_GROUPS)
    # group by area and if in working age and find the ratio to all singapore
    age_grouped = age.groupby("area").apply(
        lambda g: g.loc[g["is_working"], "Number of Residents"].sum() / g["Number of Residents"].sum()
        if g["Number of Residents"].sum() > 0 else 0
    ).reset_index()
    age_grouped.columns = ["area", "working_age_pct"]

    # AFFLUENCE SCORE
    # preprocessing data
    dwell = pd.read_excel(f"{data_dir}/dwelling.xlsx")
    dwell["area"] = dwell["Planning Area of Residence"].str.strip().str.lower()
    dwell["is_affluent"] = dwell["Dwelling Type"].str.strip().isin(AFFLUENT_DWELLINGS)
    # group by area and if in private housing and find the ratio to all of singapore
    dwell_grouped = dwell.groupby("area").apply(
        lambda g: g.loc[g["is_affluent"], "Number of Households"].sum() / g["Number of Households"].sum()
        if g["Number of Households"].sum() > 0 else 0
    ).reset_index()
    dwell_grouped.columns = ["area", "affluence_pct"]

    # COMBINE SCORES
    df = income_grouped \
        .merge(pop_total, on = "area", how = "outer") \
        .merge(age_grouped, on = "area", how = "outer") \
        .merge(dwell_grouped, on = "area", how = "outer") \
        .fillna(0)
    
    # NORMALIZE/STANDARDIZE TO ALL OF SINGAPORE 0>1
    df["income_score"] = _minmax(df["income_pct"])
    df["population_score"] = _minmax(df["total_pop"])
    df["working_age_score"] = _minmax(df["working_age_pct"])
    df["affluence_score"] = _minmax(df["affluence_pct"])

    # APPLY WEIGHTAGE AND GET IT 0>100
    df["demand_score"] = (
        df["income_score"] * WEIGHTS["income"]
        + df["population_score"] * WEIGHTS["population"]
        + df["working_age_score"] * WEIGHTS["working_age"]
        + df["affluence_score"] * WEIGHTS["affluence"]
    ) * 100

    return df[[
        "area",
        "demand_score",
        "income_score",
        "population_score",
        "working_age_score",
        "affluence_score",
    ]].sort_values("demand_score", ascending=False).reset_index(drop= True)

if __name__ == "__main__":
    scores = build_demand_scores()
    pd.set_option("display.float_format", "{:.1f}".format)
    print(scores.to_string(index= False))