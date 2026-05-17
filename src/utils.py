import pandas as pd

# ============================================================
# CLEAN TIMESTAMP
# ============================================================

def clean_timestamp(df):

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    return df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

# ============================================================
# REMOVE INF
# ============================================================

def remove_invalid(df):

    return df.replace(
        [float("inf"), float("-inf")],
        pd.NA
    )

# ============================================================
# PRINT SECTION
# ============================================================

def print_section(title):

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
