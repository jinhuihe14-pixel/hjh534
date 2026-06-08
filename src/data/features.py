import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_raw_data():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    behaviors = pd.read_csv(os.path.join(DATA_DIR, "behavior_logs.csv"))
    labels = pd.read_csv(os.path.join(DATA_DIR, "user_labels.csv"))
    return users, behaviors, labels


def build_user_features(users, behaviors, labels, window_days=7):
    behaviors["date"] = pd.to_datetime(behaviors["date"])
    latest_date = behaviors["date"].max()
    cutoff_date = latest_date - pd.Timedelta(days=window_days - 1)
    recent_behavior = behaviors[behaviors["date"] >= cutoff_date]

    agg_functions = {
        "session_duration": ["mean", "sum", "count"],
        "browse_count": ["mean", "sum"],
        "click_count": ["mean", "sum"],
        "like_count": ["mean", "sum"],
        "comment_count": ["sum"],
        "share_count": ["sum"],
    }

    stats = recent_behavior.groupby("user_id").agg(agg_functions)
    stats.columns = ["_".join(col) for col in stats.columns]
    stats = stats.reset_index()

    stats["avg_session_duration"] = stats["session_duration_mean"]
    stats["total_duration"] = stats["session_duration_sum"]
    stats["active_days_window"] = stats["session_duration_count"]
    stats["total_browse"] = stats["browse_count_sum"]
    stats["avg_browse"] = stats["browse_count_mean"]
    stats["total_click"] = stats["click_count_sum"]
    stats["total_like"] = stats["like_count_sum"]
    stats["total_interaction"] = stats["click_count_sum"] + stats["like_count_sum"] + stats["comment_count_sum"] + stats["share_count_sum"]
    stats["click_through_rate"] = np.where(stats["browse_count_sum"] > 0, stats["click_count_sum"] / stats["browse_count_sum"], 0)
    stats["like_rate"] = np.where(stats["click_count_sum"] > 0, stats["like_count_sum"] / stats["click_count_sum"], 0)

    hour_stats = recent_behavior.groupby("user_id")["active_hour"].agg(
        active_hour_mean="mean",
        active_hour_std="std"
    ).reset_index()
    stats = stats.merge(hour_stats, on="user_id", how="left")

    week_ago = latest_date - pd.Timedelta(days=7)
    two_week_ago = latest_date - pd.Timedelta(days=14)

    recent_7d = behaviors[behaviors["date"] > week_ago].groupby("user_id").size().reset_index(name="sessions_7d")
    recent_14d = behaviors[behaviors["date"] > two_week_ago].groupby("user_id").size().reset_index(name="sessions_14d")

    stats = stats.merge(recent_7d, on="user_id", how="left")
    stats = stats.merge(recent_14d, on="user_id", how="left")
    stats["sessions_7d"] = stats["sessions_7d"].fillna(0)
    stats["sessions_14d"] = stats["sessions_14d"].fillna(0)
    stats["decay_ratio"] = np.where(stats["sessions_14d"] > 0, (stats["sessions_7d"] * 2) / stats["sessions_14d"], 0)

    features = users.merge(stats, on="user_id", how="left")
    features = features.merge(labels[["user_id", "days_since_last_active", "churn_label", "is_churned_30d"]], on="user_id", how="left")

    num_cols = stats.columns.drop("user_id").tolist()
    for col in num_cols:
        if col in features.columns:
            features[col] = features[col].fillna(0)

    features["is_new_user"] = (features["register_days"] <= 30).astype(int)
    features["pay_level"] = pd.cut(
        features["total_pay_amount"],
        bins=[-1, 0, 50, 200, 500, 10000],
        labels=["0元", "0-50", "50-200", "200-500", "500+"]
    )

    features["is_churned_30d"] = features["is_churned_30d"].fillna(1).astype(int)
    return features


FEATURE_COLUMNS = [
    "register_days", "total_pay_amount", "is_vip",
    "avg_session_duration", "total_duration", "active_days_window",
    "total_browse", "avg_browse", "total_click", "total_like",
    "total_interaction", "click_through_rate", "like_rate",
    "active_hour_mean", "active_hour_std",
    "sessions_7d", "sessions_14d", "decay_ratio",
    "is_new_user",
]


def get_feature_matrix(features_df):
    df = features_df.copy()
    df["is_vip"] = df["is_vip"].astype(int)
    df["active_hour_std"] = df["active_hour_std"].fillna(df["active_hour_std"].mean())
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    X = df[FEATURE_COLUMNS].values.astype(np.float64)
    return X, FEATURE_COLUMNS


if __name__ == "__main__":
    users, behaviors, labels = load_raw_data()
    features = build_user_features(users, behaviors, labels)
    print(f"特征维度: {len(FEATURE_COLUMNS)}")
    print(f"样本数: {len(features)}")
    print(features[["user_id"] + FEATURE_COLUMNS[:5]].head())
    features.to_csv(os.path.join(DATA_DIR, "user_features.csv"), index=False)
    print("特征已保存到 data/user_features.csv")
