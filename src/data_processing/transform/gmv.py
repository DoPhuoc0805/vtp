from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.common import month_date_range
from src.utils.minio_client import (
    save_to_minio,
    extract_data_by_date,
    extract_data_by_range,
)


# =========================================================================
#  Day stage
# =========================================================================


def transform_gmv_by_day(
    date: str,
    day_prefix: str,
    raw_day_prefix: str,
    day_partition_key: str,
) -> pd.DataFrame:

    raw_df = extract_data_by_date(
        date, prefix=raw_day_prefix, day_partition_key=day_partition_key
    )

    clean_df = raw_df[
        [
            "cus_id",
            "don_ptc",
            "tong_tien",
            "tong_don",
            "thu_ho",
            "tong_cuoc_ptc",
            "don_ptc_cod",
            "thuho_tongdon",
            "tongdon_cod",
        ]
    ].copy()
    clean_df = clean_df.dropna(subset=["cus_id"])
    clean_df["cus_id"] = (
        clean_df["cus_id"]
        .apply(lambda x: x.get("member0") if isinstance(x, dict) else x)
        .astype(str)
    )
    clean_df = clean_df.rename(
        columns={"don_ptc": "count", "tong_tien": "value"}
    )

    clean_df = clean_df[
        [
            "cus_id",
            "count",
            "value",
            "tong_don",
            "thu_ho",
            "tong_cuoc_ptc",
            "don_ptc_cod",
            "thuho_tongdon",
            "tongdon_cod",
        ]
    ]

    save_to_minio(
        clean_df,
        object_name=f"{day_prefix}/{day_partition_key}={date}/data.parquet",
    )
    return clean_df


def transform_gmv_lxm(
    month: str,
    months_window: int,
    day_prefix: str,
    month_prefix: str,
    day_partition_key: str,
    month_partition_key: str,
) -> pd.DataFrame:

    start_month = (pd.Period(month, freq="M") - months_window + 1).strftime(
        "%Y%m"
    )
    start_date = month_date_range(start_month)[0]
    end_date = month_date_range(month)[1]

    day_df = extract_data_by_range(
        start_date,
        end_date,
        prefix=day_prefix,
        day_partition_key=day_partition_key,
    )
    day_df["month"] = day_df[day_partition_key].astype(str).str[:6].astype(int)

    # ---- Bước 1: Group theo (cus_id, month) ----
    agg_dict = {
        "count": "sum",
        "value": "sum",
        "tong_don": "sum",
        "thu_ho": "sum",
        "tong_cuoc_ptc": "sum",
        "don_ptc_cod": "sum",
        "thuho_tongdon": "sum",
        "tongdon_cod": "sum",
    }
    monthly = day_df.groupby(["cus_id", "month"], as_index=False)[
        list(agg_dict.keys())
    ].sum()

    # ---- Bước 2: Split COD / Non-COD ----
    cod_mask = (monthly["thu_ho"].fillna(0) > 0) | (
        monthly["thuho_tongdon"].fillna(0) > 0
    )
    group_cod = monthly[cod_mask].copy()
    group_non_cod = monthly[~cod_mask].copy()

    # ---- Bước 3: NHÓM 1 (COD) ----
    group_cod["avg_don"] = np.where(
        group_cod["thu_ho"] > 0,
        group_cod["thu_ho"] / group_cod["don_ptc_cod"].replace(0, np.nan),
        group_cod["thuho_tongdon"]
        / group_cod["tongdon_cod"].replace(0, np.nan),
    )

    group_cod["gmv"] = group_cod["count"] * group_cod["avg_don"]

    ship_rev = (
        group_cod["count"]
        * group_cod["value"]
        / group_cod["tong_don"].replace(0, np.nan)
    )
    group_cod["ship_rev_ratio"] = np.where(
        group_cod["tong_cuoc_ptc"].fillna(0) > 0,
        group_cod["tong_cuoc_ptc"] / group_cod["gmv"],
        ship_rev / group_cod["gmv"],
    )

    # ---- Bước 4: NHÓM 2 (Non-COD) ----
    med_ratio = group_cod["ship_rev_ratio"].median()

    ship_rev_non = (
        group_non_cod["count"]
        * group_non_cod["value"]
        / group_non_cod["tong_don"].replace(0, np.nan)
    )
    group_non_cod["gmv"] = np.where(
        group_non_cod["tong_cuoc_ptc"].fillna(0) > 0,
        group_non_cod["tong_cuoc_ptc"] / med_ratio,
        ship_rev_non / med_ratio,
    )

    # ---- Bước 5: Kết hợp & tính feature ----
    result = pd.concat([group_cod, group_non_cod], ignore_index=True)
    result_df = (
        result[["cus_id", "month", "gmv"]]
        .groupby("cus_id")
        .agg(**{f"f_order_gmv_l{months_window}m": ("gmv", "mean")})
        .reset_index()
    )

    save_to_minio(
        result_df,
        object_name=f"{month_prefix}/{month_partition_key}={month}/data.parquet",
    )
    return result_df
