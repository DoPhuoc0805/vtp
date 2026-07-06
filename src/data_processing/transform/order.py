from __future__ import annotations

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


def transform_order_by_day(
    date: str,
    day_prefix: str,
    raw_day_prefix: str,
    day_partition_key: str,
) -> pd.DataFrame:
    """Giai đoạn ngày: đọc raw ngày `date` từ MinIO, làm sạch, lưu xuống
    `day_prefix`/`day_partition_key`={date}/.

    -> trả về: cus_id, count, value
    """
    raw_df = extract_data_by_date(
        date, prefix=raw_day_prefix, day_partition_key=day_partition_key
    )

    clean_df = raw_df[["cus_id", "don_ptc", "tong_tien"]].copy()
    clean_df = clean_df.dropna(subset=["cus_id"])
    clean_df["cus_id"] = (
        clean_df["cus_id"]
        .apply(lambda x: x.get("member0") if isinstance(x, dict) else x)
        .astype(str)
    )
    clean_df = clean_df.rename(
        columns={"don_ptc": "count", "tong_tien": "value"}
    )

    save_to_minio(
        clean_df,
        object_name=f"{day_prefix}/{day_partition_key}={date}/data.parquet",
    )
    return clean_df


# =========================================================================
#  Core: avg count, avg value, active months
# =========================================================================


def transform_order_avg_lxm(
    month: str,
    months_window: int,
    day_prefix: str,
    month_prefix: str,
    day_partition_key: str,
    month_partition_key: str,
) -> pd.DataFrame:
    """Tính trung bình count/value theo tháng và số tháng active trong window.

    -> trả về: cus_id, f_order_avg_count_l{N}m, f_order_avg_value_l{N}m,
       f_order_active_months_l{N}m
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

    count_col = f"f_order_avg_count_l{months_window}m"
    value_col = f"f_order_avg_value_l{months_window}m"
    active_months_col = f"f_order_active_months_l{months_window}m"

    monthly_df = day_df.groupby(["cus_id", "month"]).agg(
        monthly_count=("count", "sum"), monthly_value=("value", "sum")
    )
    result_df = (
        monthly_df
        .groupby("cus_id")
        .agg(
            **{count_col: ("monthly_count", "mean")},
            **{value_col: ("monthly_value", "mean")},
            **{active_months_col: ("month", "nunique")},
        )
        .reset_index()
    )

    result_df = result_df[["cus_id", count_col, value_col, active_months_col]]

    save_to_minio(
        result_df,
        object_name=f"{month_prefix}/{month_partition_key}={month}/data.parquet",
    )
    return result_df


# =========================================================================
#  Sum: tổng doanh thu + handy 4-digit format
# =========================================================================


def transform_order_sum_handy_lxm(
    month: str,
    months_window: int,
    day_prefix: str,
    month_prefix: str,
    day_partition_key: str,
    month_partition_key: str,
) -> pd.DataFrame:
    """Tổng doanh thu (VND) trong window + phiên bản rút gọn 4-digit.

    -> trả về: cus_id, f_order_sum_value_l{N}m, f_order_sum_value_handy_l{N}m
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

    monthly_df = day_df.groupby(["cus_id", "month"]).agg(
        monthly_value=("value", "sum")
    )
    result_df = (
        monthly_df
        .groupby("cus_id")
        .agg(sum_value=("monthly_value", "sum"))
        .reset_index()
    )

    sum_col = f"f_order_sum_value_l{months_window}m"
    sum_handy_col = f"f_order_sum_value_handy_l{months_window}m"

    result_df = result_df.rename(columns={"sum_value": sum_col})

    result_df[sum_handy_col] = (
        (result_df[sum_col] / 1_000_000)
        .round()
        .clip(1, 9999)
        .astype(int)
        .astype(str)
        .str.zfill(4)
    )

    result_df = result_df[["cus_id", sum_col, sum_handy_col]]

    save_to_minio(
        result_df,
        object_name=f"{month_prefix}/{month_partition_key}={month}/data.parquet",
    )
    return result_df


# =========================================================================
#  Binning: quantile bins (equal-frequency)
# =========================================================================


def transform_order_count_bin_lxm(
    month: str,
    months_window: int,
    n_bins: int,
    day_prefix: str,
    month_prefix: str,
    day_partition_key: str,
    month_partition_key: str,
) -> pd.DataFrame:
    """Chia avg_count thành n_bins nhóm có số khách ~ bằng nhau.

    -> trả về: cus_id, f_order_avg_count_bin_l{N}m (00..n-1)
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

    monthly_df = day_df.groupby(["cus_id", "month"]).agg(
        monthly_count=("count", "sum")
    )
    avg_df = (
        monthly_df
        .groupby("cus_id")
        .agg(avg_count=("monthly_count", "mean"))
        .reset_index()
    )

    bin_col = f"f_order_avg_count_bin_l{months_window}m"
    avg_df[bin_col] = (
        pd
        .qcut(avg_df["avg_count"], q=n_bins, labels=False, duplicates="drop")
        .astype(str)
        .str.zfill(2)
    )

    result_df = avg_df[["cus_id", bin_col]]

    save_to_minio(
        result_df,
        object_name=f"{month_prefix}/{month_partition_key}={month}/data.parquet",
    )
    return result_df


# =========================================================================
#  Trend: decline streak, MoM change
# =========================================================================


def transform_order_value_decline_streak_lxm(
    month: str,
    months_window: int,
    day_prefix: str,
    month_prefix: str,
    day_partition_key: str,
    month_partition_key: str,
) -> pd.DataFrame:
    """Đếm chuỗi tháng giảm doanh thu ≥ 20% dài nhất trong window.

    Tháng thiếu được fill = 0.0001.

    -> trả về: cus_id, f_order_value_decline_streak_l{N}m (0..N-1)
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

    monthly_df = (
        day_df
        .groupby(["cus_id", "month"])
        .agg(monthly_value=("value", "sum"))
        .reset_index()
    )

    # Tạo lưới đủ months_window tháng cho mỗi cus_id
    expected_months = sorted(monthly_df["cus_id"].unique().tolist())
    if not expected_months:
        result_df = pd.DataFrame(
            columns=[
                "cus_id",
                f"f_order_value_decline_streak_l{months_window}m",
            ]
        )
        save_to_minio(
            result_df,
            object_name=f"{month_prefix}/{month_partition_key}={month}/data.parquet",
        )
        return result_df

    month_range = sorted(monthly_df["month"].unique().tolist())
    if not month_range:
        month_range = [int(start_month)]

    full_index = pd.MultiIndex.from_product(
        [expected_months, month_range], names=["cus_id", "month"]
    )
    full_df = (
        monthly_df
        .set_index(["cus_id", "month"])
        .reindex(full_index, fill_value=0.0001)
        .reset_index()
    )

    # pct_change: (prev - curr) / prev
    full_df = full_df.sort_values(["cus_id", "month"])
    full_df["prev_value"] = full_df.groupby("cus_id")["monthly_value"].shift(1)
    full_df["decline"] = (
        (full_df["prev_value"] - full_df["monthly_value"])
        / full_df["prev_value"]
    ) >= 0.2

    # Streak: đếm chuỗi True dài nhất per cus_id
    streak_col = f"f_order_value_decline_streak_l{months_window}m"

    def _max_streak(series):
        series = series.fillna(False)
        groups = (series != series.shift()).cumsum()
        streak_len = series.groupby(groups).cumcount() + 1
        return streak_len.where(series).max()

    result_df = (
        full_df
        .groupby("cus_id")["decline"]
        .apply(_max_streak)
        .fillna(0)
        .astype(int)
        .reset_index()
    )
    result_df.columns = ["cus_id", streak_col]

    save_to_minio(
        result_df,
        object_name=f"{month_prefix}/{month_partition_key}={month}/data.parquet",
    )
    return result_df


def transform_order_value_bin_lxm(
    month: str,
    months_window: int,
    n_bins: int,
    day_prefix: str,
    month_prefix: str,
    day_partition_key: str,
    month_partition_key: str,
) -> pd.DataFrame:
    """Chia avg_value thành n_bins nhóm có số khách ~ bằng nhau.

    -> trả về: cus_id, f_order_avg_value_bin_l{N}m (00..n-1)
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

    monthly_df = day_df.groupby(["cus_id", "month"]).agg(
        monthly_value=("value", "sum")
    )
    avg_df = (
        monthly_df
        .groupby("cus_id")
        .agg(avg_value=("monthly_value", "mean"))
        .reset_index()
    )

    bin_col = f"f_order_avg_value_bin_l{months_window}m"
    avg_df[bin_col] = (
        pd
        .qcut(avg_df["avg_value"], q=n_bins, labels=False, duplicates="drop")
        .astype(str)
        .str.zfill(2)
    )

    result_df = avg_df[["cus_id", bin_col]]

    save_to_minio(
        result_df,
        object_name=f"{month_prefix}/{month_partition_key}={month}/data.parquet",
    )
    return result_df
