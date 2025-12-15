# chicago_crime_choropleth_noargs_py37.py
# Python 3.7 script (no CLI args) that:
#  1) reads your processed crimes parquet from HDFS
#  2) aggregates crime counts by Community Area
#  3) downloads Chicago "Boundaries - Community Areas" GeoJSON (igwz-8jzy)
#  4) builds an interactive choropleth map (HTML)

from __future__ import print_function

import json

import pandas as pd
import requests
import folium
from folium.features import GeoJsonTooltip
from pyspark.sql import SparkSession, functions as F

PARQUET_PATH = "hdfs://namenode:9000/data/processed/chicago_crimes_clean.parquet"

GEOJSON_URL = "https://data.cityofchicago.org/resource/igwz-8jzy.geojson"

OUT_HTML = "chicago_crimes_choropleth.html"
MAP_TILES = "cartodbpositron"


def build_spark():
    spark = (SparkSession.builder
                        .appName("ChicagoCrimeChoropleth")
                        .master("spark://spark-master:7077")
                        .getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def fetch_geojson():
    headers = {"User-Agent": "Mozilla/5.0 (compatible; bigdata-project/1.0)"}
    r = requests.get(GEOJSON_URL, headers=headers, timeout=60)
    r.raise_for_status()
    return r.json()


def infer_area_key(geojson_obj):
    """
    Find the property field containing the Community Area number.
    Usually one of: area_numbe / area_num_1 / area_num
    """
    props = geojson_obj["features"][0].get("properties", {})
    keys = list(props.keys())

    candidates = ["area_numbe", "area_num_1", "area_num", "comarea", "community_area"]
    for c in candidates:
        if c in keys:
            return c

    # fallback heuristic
    for k in keys:
        lk = k.lower()
        if ("area" in lk) and (("num" in lk) or ("no" in lk)):
            return k

    raise ValueError("Could not infer Community Area key from GeoJSON properties keys: {0}".format(keys))


def infer_name_key(geojson_obj):
    """
    Find the community area name field (often 'community').
    """
    props = geojson_obj["features"][0].get("properties", {})
    keys = list(props.keys())
    for k in ["community", "name", "community_area_name"]:
        if k in keys:
            return k
    return None


def aggregate_counts(spark):
    df = spark.read.parquet(PARQUET_PATH)

    # Use crime_year if present (else Year)
    year_col = "crime_year" if "crime_year" in df.columns else "Year"

    base = (
        df.select(
            F.col("Community Area").cast("int").alias("community_area"),
            F.col(year_col).cast("int").alias("year"),
        )
        .where(F.col("community_area").isNotNull())
    )

    counts = (
        base.groupBy("community_area")
        .agg(F.count(F.lit(1)).alias("crime_count"))
        .orderBy(F.col("crime_count").desc())
    )

    return counts.toPandas()


def add_counts_to_geojson(geojson_obj, area_key, counts_pd):
    """
    Adds 'crime_count' into feature.properties so tooltips can show it.
    """
    lookup = {}
    for _, row in counts_pd.iterrows():
        if pd.notna(row["community_area"]):
            lookup[int(row["community_area"])] = int(row["crime_count"])

    for feat in geojson_obj.get("features", []):
        props = feat.setdefault("properties", {})
        raw = props.get(area_key)

        try:
            area_id = int(raw)
        except Exception:
            area_id = None

        props["crime_count"] = lookup.get(area_id, 0)

    return geojson_obj


def compute_bins(counts_pd):
    """
    5-class legend like the screenshot (quantile bins).
    """
    s = counts_pd["crime_count"].astype(float)
    qs = s.quantile([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
    # ensure strictly increasing bins (folium requires it)
    bins = sorted(list(set([int(round(x)) for x in qs])))
    if len(bins) < 3:
        # fallback
        mn = int(s.min())
        mx = int(s.max())
        step = max(1, int((mx - mn) / 5.0))
        bins = [mn, mn + step, mn + 2 * step, mn + 3 * step, mn + 4 * step, mx]
    return bins


def build_map(geojson_obj, area_key, name_key, counts_pd):
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=10, tiles=MAP_TILES)

    bins = compute_bins(counts_pd)

    folium.Choropleth(
        geo_data=geojson_obj,
        data=counts_pd,
        columns=["community_area", "crime_count"],
        key_on="feature.properties.{0}".format(area_key),
        fill_color="YlOrRd",
        fill_opacity=0.75,
        line_opacity=0.25,
        nan_fill_opacity=0.2,
        bins=bins,
        legend_name="Crimes - 2001 to Present",
    ).add_to(m)

    tooltip_fields = []
    tooltip_aliases = []

    if name_key:
        tooltip_fields.append(name_key)
        tooltip_aliases.append("Area")

    tooltip_fields.extend([area_key, "crime_count"])
    tooltip_aliases.extend(["Community Area #", "Crimes"])

    folium.GeoJson(
        geojson_obj,
        name="Community Areas",
        tooltip=GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases, localize=True),
        style_function=lambda _: {"fillOpacity": 0.0, "weight": 0.6},
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def main():
    print("Reading parquet from:", PARQUET_PATH)
    spark = build_spark()

    counts_pd = aggregate_counts(spark)
    print("Aggregated community areas:", len(counts_pd))

    print("Loading boundaries from:", GEOJSON_URL)
    geo = fetch_geojson()

    area_key = infer_area_key(geo)
    name_key = infer_name_key(geo)
    print("GeoJSON keys -> area:", area_key, "| name:", name_key)

    geo = add_counts_to_geojson(geo, area_key, counts_pd)

    m = build_map(geo, area_key, name_key, counts_pd)
    m.save(OUT_HTML)
    print("Saved map HTML to:", OUT_HTML)


if __name__ == "__main__":
    main()
