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
    """Giai đoạn ngày: trích xuất các cột cần cho tính GMV.

    -> trả về: cus_id, count, value, tong_don, thu_ho
    """
    raw_df = extract_data_by_date(
        date, prefix=raw_day_prefix, day_partition_key=day_partition_key
    )

    clean_df = raw_df[
        ["cus_id", "don_ptc", "tong_tien", "tong_don", "thu_ho"]
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

    clean_df = clean_df[["cus_id", "count", "value", "tong_don", "thu_ho"]]

    save_to_minio(
        clean_df,
        object_name=f"{day_prefix}/{day_partition_key}={date}/data.parquet",
    )
    return clean_df


# =========================================================================
#  Month stage: GMV estimate
# =========================================================================


def transform_gmv_lxm(
    month: str,
    months_window: int,
    day_prefix: str,
    month_prefix: str,
    day_partition_key: str,
    month_partition_key: str,
) -> pd.DataFrame:
    """Tính GMV ước tính cho cả COD và Non-COD trong window.

    COD (thu_ho > 0): GMV = thu_ho trực tiếp.
    Non-COD (thu_ho NaN hoặc 0): suy từ ship_rev / med_ratio,
    trong đó med_ratio = median(ship_rev / thu_ho) từ toàn bộ COD.

    -> trả về: cus_id, f_order_gmv_l{N}m
    """
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

    # ---- Bước 1: Tính ship_rev ----
    # Non-COD: don_ptc NaN → dùng tong_don
    effective_count = day_df["count"].fillna(day_df["tong_don"]).to_numpy()
    ship_rev = np.where(
        day_df["tong_don"].notna() & (day_df["tong_don"] > 0),
        effective_count * day_df["value"].to_numpy() / day_df["tong_don"].to_numpy(),
        np.nan,
    )

    # ---- Bước 2: median ship_rev_ratio từ COD ----
    cod_mask = day_df["thu_ho"].notna() & (day_df["thu_ho"] > 0)
    if cod_mask.any():
        cod_ratio = ship_rev[cod_mask] / day_df.loc[cod_mask, "thu_ho"].to_numpy()
        med_ratio = np.nanmedian(cod_ratio)
    else:
        med_ratio = 1.0

    # ---- Bước 3: Tính GMV 1 lần ----
    gmv = np.where(
        cod_mask,
        day_df["thu_ho"].fillna(0.0).to_numpy(),
        np.where(~np.isnan(ship_rev), ship_rev / med_ratio, 0.0),
    )
    day_df["gmv"] = gmv

    # ---- Bước 4: Tổng hợp theo tháng ----
    monthly_df = (
        day_df
        .groupby(["cus_id", "month"])
        .agg(monthly_gmv=("gmv", "sum"))
    )
    result_df = (
        monthly_df
        .groupby("cus_id")
        .agg(**{f"f_order_gmv_l{months_window}m": ("monthly_gmv", "mean")})
        .reset_index()
    )

    save_to_minio(
        result_df,
        object_name=f"{month_prefix}/{month_partition_key}={month}/data.parquet",
    )
    return result_df
