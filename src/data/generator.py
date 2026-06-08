import os
import random
import hashlib
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

CONTENT_CATEGORIES = ["科技", "娱乐", "体育", "财经", "教育", "生活", "游戏", "美食", "旅游", "时尚"]
USER_TYPES = ["高价值付费", "普通付费", "活跃免费", "轻度沉睡", "深度流失"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
CITIES = ["一线城市", "新一线", "二线城市", "三线及以下"]


def _hash_id(uid):
    return hashlib.md5(str(uid).encode()).hexdigest()[:8]


def generate_users(n=1000):
    users = []
    for i in range(n):
        uid = f"user_{i:05d}"
        age_group = random.choices(AGE_GROUPS, weights=[0.25, 0.35, 0.2, 0.12, 0.08])[0]
        city = random.choices(CITIES, weights=[0.2, 0.3, 0.25, 0.25])[0]
        register_days = random.randint(30, 720)
        total_pay = round(max(0, np.random.normal(200, 300)), 2)
        is_vip = total_pay > 100
        prefer_categories = random.sample(CONTENT_CATEGORIES, k=random.randint(2, 5))
        users.append({
            "user_id": uid,
            "user_hash": _hash_id(uid),
            "age_group": age_group,
            "city_level": city,
            "register_days": register_days,
            "total_pay_amount": total_pay,
            "is_vip": is_vip,
            "prefer_categories": ",".join(prefer_categories),
        })
    df = pd.DataFrame(users)
    df.to_csv(os.path.join(DATA_DIR, "users.csv"), index=False)
    return df


def generate_behavior_logs(users_df, days=30):
    records = []
    end_date = datetime.now()
    user_list = users_df.to_dict("records")
    for user in user_list:
        is_churned = random.random() < 0.2
        is_light_sleep = (not is_churned) and random.random() < 0.3
        base_active_days = random.randint(15, 28) if not is_churned and not is_light_sleep else (
            random.randint(5, 14) if is_light_sleep else random.randint(0, 3)
        )
        active_days = min(days, max(0, base_active_days))
        active_day_set = set(random.sample(range(days), k=active_days))
        for day in range(days):
            date = end_date - timedelta(days=days - day - 1)
            date_str = date.strftime("%Y-%m-%d")
            if day in active_day_set:
                sessions = random.randint(1, 6) if not is_churned else 1
                for _ in range(sessions):
                    session_start = date.replace(hour=random.randint(7, 23), minute=random.randint(0, 59))
                    duration = random.randint(60, 3600) if not is_churned else random.randint(30, 300)
                    browse_count = random.randint(5, 50) if not is_churned else random.randint(1, 5)
                    click_count = int(browse_count * random.uniform(0.1, 0.4))
                    like_count = int(click_count * random.uniform(0.2, 0.6))
                    comment_count = random.randint(0, 3) if random.random() < 0.3 else 0
                    share_count = random.randint(0, 2) if random.random() < 0.15 else 0
                    records.append({
                        "user_id": user["user_id"],
                        "date": date_str,
                        "session_duration": duration,
                        "browse_count": browse_count,
                        "click_count": click_count,
                        "like_count": like_count,
                        "comment_count": comment_count,
                        "share_count": share_count,
                        "active_hour": session_start.hour,
                    })
    df = pd.DataFrame(records)
    df.to_csv(os.path.join(DATA_DIR, "behavior_logs.csv"), index=False)
    return df


def generate_churn_labels(users_df, behavior_df):
    last_active = behavior_df.groupby("user_id")["date"].max().reset_index()
    last_active.columns = ["user_id", "last_active_date"]
    last_active["last_active_date"] = pd.to_datetime(last_active["last_active_date"])
    today = pd.Timestamp.now().normalize()
    last_active["days_since_last_active"] = (today - last_active["last_active_date"]).dt.days
    merged = users_df.merge(last_active, on="user_id", how="left")
    merged["days_since_last_active"] = merged["days_since_last_active"].fillna(999)
    def _label(row):
        days = row["days_since_last_active"]
        if days <= 3:
            return "活跃"
        elif days <= 7:
            return "轻度沉睡"
        else:
            return "深度流失"
    merged["churn_label"] = merged.apply(_label, axis=1)
    merged["is_churned_30d"] = (merged["churn_label"] == "深度流失").astype(int)
    merged.to_csv(os.path.join(DATA_DIR, "user_labels.csv"), index=False)
    return merged


def generate_all_data(n_users=1000, days=30):
    print(f"生成用户数据: {n_users} 个用户...")
    users = generate_users(n_users)
    print(f"生成行为日志: {days} 天...")
    behaviors = generate_behavior_logs(users, days)
    print("生成流失标签...")
    labels = generate_churn_labels(users, behaviors)
    print("数据生成完成！")
    print(f"  - 用户数: {len(users)}")
    print(f"  - 行为记录: {len(behaviors)}")
    print(f"  - 流失分布: {labels['churn_label'].value_counts().to_dict()}")
    return users, behaviors, labels


if __name__ == "__main__":
    generate_all_data()
