from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    BooleanType,
    StringType,
)
from pyspark.sql.functions import udf

RAW_PATH = "hdfs://namenode:9000/data/raw/chicago_crimes.parquet"
PROCESSED_PATH = "hdfs://namenode:9000/data/processed/chicago_crimes_clean.parquet"

def build_spark():
    spark = (
        SparkSession.builder
        .appName("ChicagoCrimeETL")
        .master("spark://spark-master:7077")
        # bigger partition, more parallel tasks, less memory per task
        .config("spark.sql.shuffle.partitions", "64")
        # prevent small filesize causing HDFS small files problem
        .config("spark.sql.files.maxRecordsPerFile", "500000")
        .config("spark.executor.memory", "3g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_raw(spark):
    df = spark.read.parquet(RAW_PATH)
    return df


def clean_data(df):
    # Drop rows with missing core fields
    required_cols = ["ID", "Date", "Primary Type", "Year"]
    df = df.dropna(subset=required_cols)

    # Deduplicate by ID
    df = df.dropDuplicates(["ID"])

    # Cast numeric columns
    numeric_int_cols = ["Beat", "District", "Ward", "Community Area", "Year"]
    numeric_double_cols = ["X Coordinate", "Y Coordinate", "Latitude", "Longitude"]

    for c in numeric_int_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(IntegerType()))

    for c in numeric_double_cols:
        if c in df.columns:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))

    # Normalize boolean-ish columns
    for bool_col in ["Arrest", "Domestic"]:
        if bool_col in df.columns:
            df = df.withColumn(
                bool_col,
                F.when(F.lower(F.col(bool_col)) == "true", True)
                .when(F.lower(F.col(bool_col)) == "false", False)
                .otherwise(None)
                .cast(BooleanType()),
            )

    return df


def add_time_features(df):
    df = df.withColumn(
        "crime_datetime",
        F.coalesce(
            F.to_timestamp(F.col("Date"), "MM/dd/yyyy hh:mm:ss a"),
            F.to_timestamp(F.col("Date"), "yyyy-MM-dd'T'HH:mm:ss.SSS"),
            F.to_timestamp(F.col("Date"), "yyyy-MM-dd'T'HH:mm:ss"),
        ),
    )

    df = df.dropna(subset=["crime_datetime"])

    df = (
        df.withColumn("crime_year", F.year("crime_datetime"))
          .withColumn("crime_month", F.month("crime_datetime"))
          .withColumn("crime_day", F.dayofmonth("crime_datetime"))
          .withColumn("crime_hour", F.hour("crime_datetime"))
          .withColumn("crime_day_of_week", F.dayofweek("crime_datetime"))
    )

    df = df.withColumn(
        "Year",
        F.when(F.col("crime_year").isNotNull(), F.col("crime_year"))
         .otherwise(F.col("Year"))
         .cast(IntegerType()),
    )

    df = df.withColumn(
        "time_of_day",
        F.when((F.col("crime_hour") >= 5) & (F.col("crime_hour") < 12), "MORNING")
         .when((F.col("crime_hour") >= 12) & (F.col("crime_hour") < 17), "AFTERNOON")
         .when((F.col("crime_hour") >= 17) & (F.col("crime_hour") < 22), "EVENING")
         .otherwise("NIGHT"),
    )

    df = df.withColumn(
        "season",
        F.when(F.col("crime_month").isin(12, 1, 2), "WINTER")
         .when(F.col("crime_month").isin(3, 4, 5), "SPRING")
         .when(F.col("crime_month").isin(6, 7, 8), "SUMMER")
         .otherwise("FALL"),
    )
    return df


# ---------- Location category ----------
# Residential (house, appartment...)
# Transportation (roads, transit, vehicles, airports...)
# Food & Entertainment (bar, casino, club...)
# Commercial (retail, offices, services, banks...)
# Institutional (schools, hospitals, govt, police, fire...)
# Public Outdoor (streets, parks...)
# Other
def _location_category(loc: str) -> str:
    if not loc:
        return "Other"
    s = loc.upper()

    # Residential
    if (
        "RESIDENCE" in s
        or "APARTMENT" in s
        or "HOUSE" in s
        or "ROOMING HOUSE" in s
        or "COACH HOUSE" in s
        or "PORCH" in s
        or "HALLWAY" in s
        or "GARAGE" in s
        or "YARD" in s
        or "BASEMENT" in s
    ):
        return "Residential"

    # Transportation (roads, transit, vehicles, airports)
    if (
        "CTA" in s
        or "TRAIN" in s
        or "BUS" in s
        or "RAILROAD" in s
        or "HIGHWAY" in s
        or "EXPRESSWAY" in s
        or "AIRPORT" in s
        or "AIRCRAFT" in s
        or "VEHICLE" in s
        or "TRUCK" in s
        or "TAXI" in s
        or "LIVERY" in s
        or "PARKING LOT" in s
        or "PARKING LOT/GARAGE" in s
    ):
        return "Transportation"

    # Food & Entertainment
    if (
        "BAR" in s
        or "TAVERN" in s
        or "LIQUOR STORE" in s
        or "RESTAURANT" in s
        or "MOVIE HOUSE" in s
        or "THEATER" in s
        or "CASINO" in s
        or "CLUB" in s
        or "SPORTS ARENA" in s
        or "STADIUM" in s
        or "YMCA" in s
        or "BOWLING ALLEY" in s
        or "POOL ROOM" in s
        or "POOLROOM" in s
        or "BANQUET HALL" in s
    ):
        return "Food & Entertainment"

    # Commercial (retail, offices, services, banks)
    if (
        "STORE" in s
        or "SHOP" in s
        or "DEALERSHIP" in s
        or "BANK" in s
        or "CREDIT UNION" in s
        or "CURRENCY EXCHANGE" in s
        or "PAWN" in s
        or "OFFICE" in s
        or "NEWSSTAND" in s
        or "CLEANERS" in s
        or "LAUNDROMAT" in s
        or "BARBER" in s
        or "BEAUTY SALON" in s
        or "CAR WASH" in s
        or "COIN OPERATED MACHINE" in s
    ):
        return "Commercial"

    # Institutional (schools, hospitals, govt, religious, police, fire)
    if (
        "SCHOOL" in s
        or "COLLEGE" in s
        or "UNIVERSITY" in s
        or "HOSPITAL" in s
        or "NURSING" in s
        or "RETIREMENT HOME" in s
        or "GOVERNMENT" in s
        or "POLICE" in s
        or "JAIL" in s
        or "PRISON" in s
        or "CHURCH" in s
        or "SYNAGOGUE" in s
        or "PLACE OF WORSHIP" in s
        or "LIBRARY" in s
        or "FIRE STATION" in s
        or "PUBLIC HIGH SCHOOL" in s
        or "PUBLIC GRAMMAR SCHOOL" in s
    ):
        return "Institutional"

    # Industrial (factories, warehouses, construction, yards)
    if (
        "FACTORY" in s
        or "WAREHOUSE" in s
        or "CONSTRUCTION" in s
        or "JUNK YARD" in s
        or "GARBAGE DUMP" in s
        or "TRUCKING TERMINAL" in s
        or "FARM" in s
        or "INDUSTRIAL" in s
        or "LOADING DOCK" in s
    ):
        return "Industrial"

    # Public Outdoor (streets, parks, natural areas, vacant lots)
    if (
        "STREET" in s
        or "SIDEWALK" in s
        or "ALLEY" in s
        or "PARK" in s
        or "PARK PROPERTY" in s
        or "BEACH" in s
        or "LAKEFRONT" in s
        or "LAKE" in s
        or "RIVERBANK" in s
        or "RIVER BANK" in s
        or "RIVER" in s
        or "FOREST PRESERVE" in s
        or "PRAIRIE" in s
        or "VACANT LOT" in s
        or "VACANT LOT/LAND" in s
        or "VACANT LOT / LAND" in s
        or "WOODED AREA" in s
        or "BRIDGE" in s
        or "GANGWAY" in s
        or "PLAY LOT" in s
        or "CHA GROUNDS" in s
        or "YARD" in s
        or "EXPRESSWAY EMBANKMENT" in s
    ):
        return "Public Outdoor"

    return "Other"

def _location_weight(category: str) -> int:
    if category == "Public Outdoor":
        return 3
    if category == "Transportation":
        return 3
    if category == "Food & Entertainment":
        return 3
    if category in ("Residential", "Commercial", "Institutional", "Industrial"):
        return 2
    return 1


location_category_udf = udf(_location_category, StringType())
location_weight_udf = udf(_location_weight, IntegerType())


# ---------- Primary Type category & severity UDFs ----------

VIOLENT_TYPES = {
    "HOMICIDE",
    "CRIM SEXUAL ASSAULT",
    "CRIMINAL SEXUAL ASSAULT",
    "SEX OFFENSE",
    "BATTERY",
    "ASSAULT",
    "ROBBERY",
    "KIDNAPPING",
    "OFFENSE INVOLVING CHILDREN",
    "HUMAN TRAFFICKING",
    "DOMESTIC VIOLENCE",
    "INTIMIDATION",
    "STALKING",
}

PROPERTY_TYPES = {
    "BURGLARY",
    "THEFT",
    "MOTOR VEHICLE THEFT",
    "ARSON",
    "CRIMINAL DAMAGE",
    "CRIMINAL TRESPASS",
    "DECEPTIVE PRACTICE",
    "OTHER OFFENSE",
    "OBSCENITY",
}

DRUG_WEAPON_TYPES = {
    "NARCOTICS",
    "OTHER NARCOTIC VIOLATION",
    "WEAPONS VIOLATION",
    "CONCEALED CARRY LICENSE VIOLATION",
}

PUBLIC_ORDER_TYPES = {
    "PUBLIC PEACE VIOLATION",
    "INTERFERENCE WITH PUBLIC OFFICER",
    "LIQUOR LAW VIOLATION",
    "GAMBLING",
    "PUBLIC INDECENCY",
    "RITUALISM",
    "NON-CRIMINAL",
}


def _crime_category(pt: str) -> str:
    if not pt:
        return "Other Crime"
    s = pt.upper()
    if s in VIOLENT_TYPES:
        return "Violent Crime"
    if s in PROPERTY_TYPES:
        return "Property Crime"
    if s in DRUG_WEAPON_TYPES:
        return "Drug/Weapon Crime"
    if s in PUBLIC_ORDER_TYPES:
        return "Public Order Crime"
    return "Other Crime"


# Severity: Level 3 (10) > Level 2 (3) > Level 1 (1)
HIGH_SEVERITY = {
    "HOMICIDE",
    "CRIM SEXUAL ASSAULT",
    "CRIMINAL SEXUAL ASSAULT",
    "SEX OFFENSE",
    "KIDNAPPING",
    "HUMAN TRAFFICKING",
    "ARSON",
    "OFFENSE INVOLVING CHILDREN",
}

MEDIUM_SEVERITY = VIOLENT_TYPES | DRUG_WEAPON_TYPES | {
    "BURGLARY",
    "ROBBERY",
    "MOTOR VEHICLE THEFT",
    "CRIMINAL DAMAGE",
    "CRIMINAL TRESPASS",
    "DECEPTIVE PRACTICE",
    "PROSTITUTION",
}


def _crime_severity_level(pt: str) -> str:
    if not pt:
        return "Level 1"
    s = pt.upper()
    if s in HIGH_SEVERITY:
        return "Level 3"
    if s in MEDIUM_SEVERITY:
        return "Level 2"
    return "Level 1"


def _crime_severity_weight(pt: str) -> int:
    if not pt:
        return 1
    s = pt.upper()
    if s in HIGH_SEVERITY:
        return 10
    if s in MEDIUM_SEVERITY:
        return 3
    return 1


crime_category_udf = udf(_crime_category, StringType())
crime_severity_level_udf = udf(_crime_severity_level, StringType())
crime_severity_weight_udf = udf(_crime_severity_weight, IntegerType())


def add_crime_features(df):
    # Location category + weight
    df = df.withColumn(
        "location_category",
        location_category_udf(F.col("Location Description")),
    )
    df = df.withColumn(
        "location_risk_weight",
        location_weight_udf(F.col("location_category")).cast(DoubleType()),
    )

    # Crime category + severity
    df = df.withColumn(
        "crime_category",
        crime_category_udf(F.col("Primary Type")),
    )
    df = df.withColumn(
        "crime_severity_level",
        crime_severity_level_udf(F.col("Primary Type")),
    )
    df = df.withColumn(
        "crime_severity_weight",
        crime_severity_weight_udf(F.col("Primary Type")).cast(DoubleType()),
    )

    # Flags
    df = df.withColumn(
        "is_violent",
        (F.col("crime_category") == "Violent Crime"),
    )

    df = df.withColumn(
        "is_night",
        (F.col("crime_hour") >= 22) | (F.col("crime_hour") < 4),
    )

    # Arrest / domestic aliases, default False if missing
    if "Arrest" in df.columns:
        df = df.withColumn("was_arrest_made", F.col("Arrest"))
    else:
        df = df.withColumn("was_arrest_made", F.lit(False))

    if "Domestic" in df.columns:
        df = df.withColumn("is_domestic_incident", F.col("Domestic"))
    else:
        df = df.withColumn("is_domestic_incident", F.lit(False))

    # Public space flag (non-residential outdoor / transit / nightlife)
    df = df.withColumn(
        "is_public_space",
        (F.col("location_category") == "Public Outdoor")
        | (F.col("location_category") == "Transportation")
        | (F.col("location_category") == "Food & Entertainment"),
    )

    # Composite risk score
    df = df.withColumn(
        "risk_score_raw",
        F.col("crime_severity_weight")
        + F.col("location_risk_weight")
        + F.when(F.col("is_night"), 2.0).otherwise(0.0)
        + F.when(F.col("is_domestic_incident"), 1.5).otherwise(0.0)
        + F.when(F.col("is_public_space"), 1.0).otherwise(0.0),
    )

    # ---- Normalize risk_score_raw to [1, 10] ----
    #  min/max given the weights:
    #  min_raw = 1 (crime lv1) + 1 (location low) = 2.0
    #  max_raw = 10 (crime lv3) + 3 (location high) + 2 + 1.5 + 1 = 17.5
    min_raw = 2.0
    max_raw = 17.5
    scale = 9.0 / (max_raw - min_raw)  # 1..10 range

    df = df.withColumn(
        "risk_score",
        1.0 + (F.col("risk_score_raw") - F.lit(min_raw)) * F.lit(scale)
    )

    # Clamp just in case any future changes push it outside [1, 10]
    df = df.withColumn(
        "risk_score",
        F.when(F.col("risk_score") < 1.0, 1.0)
         .when(F.col("risk_score") > 10.0, 10.0)
         .otherwise(F.col("risk_score"))
    )

    return df


def write_processed(df):
    """
    Prune to important columns, then write partitioned by Year.
    This reduces row width during the final shuffle to avoid OOM.
    """
    base_cols = [
        "ID",
        "Case Number",
        "Date",
        "Primary Type",
        "Description",
        "Location Description",
        "Arrest",
        "Domestic",
        "Beat",
        "District",
        "Ward",
        "Community Area",
        "Year",
        "Latitude",
        "Longitude",
    ]

    feature_cols = [
        "crime_datetime",
        "crime_year",
        "crime_month",
        "crime_day",
        "crime_hour",
        "crime_day_of_week",
        "time_of_day",
        "season",
        "location_category",
        "location_risk_weight",
        "crime_category",
        "crime_severity_level",
        "crime_severity_weight",
        "is_violent",
        "is_night",
        "was_arrest_made",
        "is_domestic_incident",
        "is_public_space",
        "risk_score",
    ]

    cols_to_keep = [c for c in (base_cols + feature_cols) if c in df.columns]
    df_out = df.select(*cols_to_keep)

    (
        df_out.write.mode("overwrite")
        .partitionBy("Year")
        .parquet(PROCESSED_PATH)
    )


def main():
    spark = build_spark()

    print("=== Loading raw data from HDFS ===")
    raw_df = load_raw(spark)
    print(f"Raw count: {raw_df.count()}")

    print("=== Cleaning data ===")
    clean_df = clean_data(raw_df)

    print("=== Adding time features ===")
    time_df = add_time_features(clean_df)

    print("=== Adding crime features ===")
    feat_df = add_crime_features(time_df)

    print("=== Writing processed data to HDFS ===")
    write_processed(feat_df)

    print("Done. Processed dataset written to:", PROCESSED_PATH)
    spark.stop()


if __name__ == "__main__":
    main()
