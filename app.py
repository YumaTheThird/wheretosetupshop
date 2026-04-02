import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from demand_score import build_demand_scores
from scoring import (
    score_area_df, apply_decision_tree,
    area_to_dict, safe_float, TREE_WEIGHTS,
)

app = Flask(__name__)
BASE = os.path.dirname(__file__)

def _load():
    estab = pd.read_excel(os.path.join(BASE, "data", "NEA_Establishments.xlsx"))
    estab["PLN_AREA_N"] = estab["PLN_AREA_N"].str.strip().str.title()
    estab["Region"] = estab["Region"].str.strip()
    estab["Type"] = estab["Type"].str.strip()

    pa = pd.read_excel(os.path.join(BASE, "data", "planning_area_combined.xlsx"))
    pa["planning_area_name"] = pa["planning_area_name"].str.strip()
    pa["region"] = pa["region"].str.strip()
    area = pa[pa["quarter"] == pa["quarter"].max()].copy().reset_index(drop=True)

    demand = build_demand_scores(os.path.join(BASE, "data", "demographics"))
    demand["area_key"] = demand["area"].str.strip().str.title()
    area["area_key"] = area["planning_area_name"].str.title()
    area = area.merge(
        demand[["area_key", "demand_score", "income_score",
                "population_score", "working_age_score", "affluence_score"]],
        on="area_key", how="left",
    )
    area["demand_score"] = area["demand_score"].fillna(area["demand_score"].median())
    return estab, area

ESTAB, AREA = _load()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recommender")
def recommender():
    return render_template("recommender.html")

@app.route("/api/overview_stats")
def overview_stats():
    return jsonify({
        "total_establishments": int(len(ESTAB)),
        "planning_areas": int(len(AREA)),
        "regions": int(AREA["region"].nunique()),
        "fnb_types": int(ESTAB["Type"].nunique()),
        "highest_demand_area": AREA.loc[AREA["demand_score"].idxmax(), "planning_area_name"],
        "lowest_rent_area": AREA.loc[AREA["avg_median_rent_psm"].idxmin(), "planning_area_name"],
        "most_accessible_area": AREA.loc[
            (AREA["avg_mrt_exits_500m"].fillna(0) + AREA["avg_bus_stops_500m"].fillna(0)).idxmax(),
            "planning_area_name"],
    })

@app.route("/api/filters")
def get_filters():
    return jsonify({
        "rent_range": {"min": safe_float(AREA["avg_median_rent_psm"].min()),
                       "max": safe_float(AREA["avg_median_rent_psm"].max())},
        "fnb_types": sorted(ESTAB["Type"].dropna().unique().tolist()),
        "regions": sorted(AREA["region"].dropna().unique().tolist()),
        "planning_areas": sorted(AREA["planning_area_name"].unique().tolist()),
        "tree_weights": TREE_WEIGHTS,
    })

@app.route("/api/recommend")
def recommend():
    min_rent = float(request.args.get("min_rent", 0))
    max_rent = float(request.args.get("max_rent", 9999))
    fnb_type = request.args.get("fnb_type", "").strip()
    region = request.args.get("region", "").strip()
    top_n = min(int(request.args.get("top_n", 5)), 20)

    filtered = apply_decision_tree(AREA, min_rent, max_rent, region)
    if filtered.empty:
        return jsonify({"results": [], "message": "No areas match your rental budget."})

    scored = score_area_df(filtered)
    ranked = scored.sort_values("final_score", ascending=False).head(top_n)

    results = []
    for _, row in ranked.iterrows():
        area_name = row["planning_area_name"]
        mask = ESTAB["PLN_AREA_N"].str.lower() == area_name.lower()
        estabs = ESTAB[mask].copy()
        if fnb_type:
            estabs = estabs[estabs["Type"] == fnb_type]
        entry = area_to_dict(row)
        entry["type_breakdown"] = ESTAB[mask]["Type"].value_counts().head(5).to_dict()
        entry["sample_locations"] = estabs[["licensee_name", "premises_address", "Type",
            "latitude", "longitude", "mrt_exits_500m", "bus_stops_500m"]].head(6).to_dict("records")
        entry["fnb_count_filtered"] = len(estabs)
        results.append(entry)

    return jsonify({"results": results,
                    "filters_applied": {"rent_range": [min_rent, max_rent],
                                        "fnb_type": fnb_type or "all",
                                        "region": region or "all"},
                    "tree_weights": TREE_WEIGHTS})

@app.route("/api/area/<area_name>")
def area_detail(area_name):
    match = AREA[AREA["planning_area_name"].str.lower() == area_name.lower()]
    if match.empty:
        return jsonify({"error": f"Area '{area_name}' not found"}), 404
    scored = score_area_df(match)
    result = area_to_dict(scored.iloc[0])
    estabs = ESTAB[ESTAB["PLN_AREA_N"].str.lower() == area_name.lower()]
    result["all_type_counts"] = estabs["Type"].value_counts().to_dict()
    result["total_estab_in_db"] = len(estabs)
    return jsonify(result)

@app.route("/api/establishments")
def establishments():
    area = request.args.get("area", "").strip()
    fnb_type = request.args.get("fnb_type", "").strip()
    region = request.args.get("region", "").strip()
    page = max(int(request.args.get("page", 1)), 1)
    per_page = min(int(request.args.get("per_page", 20)), 100)
    df = ESTAB.copy()
    if area:
        df = df[df["PLN_AREA_N"].str.lower() == area.lower()]
    if region:
        df = df[df["Region"].str.lower() == region.lower()]
    if fnb_type:
        df = df[df["Type"] == fnb_type]
    total = len(df)
    start = (page - 1) * per_page
    return jsonify({"total": total, "page": page, "per_page": per_page,
                    "pages": (total + per_page - 1) // per_page,
                    "data": df.iloc[start:start + per_page][["licensee_name", "premises_address",
                        "Type", "PLN_AREA_N", "Region", "latitude", "longitude",
                        "mrt_exits_500m", "bus_stops_500m"]].to_dict("records")})

if __name__ == "__main__":
    app.run(debug=True, port=5000)