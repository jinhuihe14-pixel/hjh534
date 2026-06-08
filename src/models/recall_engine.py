import os
import sys
import json
import random
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")


CONTENT_CATEGORIES = ["科技", "娱乐", "体育", "财经", "教育", "生活", "游戏", "美食", "旅游", "时尚"]

COUPON_TYPES = {
    "vip_discount": {"name": "会员折扣券", "value": "7折", "cost": 15},
    "cash_coupon": {"name": "现金券", "value": "10元", "cost": 10},
    "points_bonus": {"name": "积分奖励", "value": "200积分", "cost": 5},
    "exclusive_content": {"name": "专属内容", "value": "7天免费", "cost": 8},
    "no_coupon": {"name": "无优惠券", "value": "仅内容推荐", "cost": 0},
}

PUSH_TEMPLATES = {
    "content": [
        "🔥 你感兴趣的{category}又上新了，快来看看！",
        "📌 精选{category}内容，不容错过！",
        "✨ {category}热门TOP10，点击查看",
    ],
    "coupon": [
        "🎁 专属福利已到账，{coupon_value}等你来领！",
        "💰 送你{coupon_value}，限时使用！",
        "🎉 惊喜好礼：{coupon_value}，点击领取",
    ],
    "comeback": [
        "👋 好久不见，有新内容等你发现",
        "🌟 离开的日子，精彩不断更新",
        "💫 欢迎回来，首页为你准备了惊喜",
    ],
}

BEST_PUSH_HOURS = {
    "早间活跃": [7, 8, 9],
    "午间活跃": [11, 12, 13],
    "晚间活跃": [19, 20, 21],
    "夜间活跃": [22, 23],
    "通用时段": [12, 18, 20],
}


class RecallStrategyEngine:
    def __init__(self):
        self.strategies = self._define_strategies()

    def _define_strategies(self):
        return {
            "高价值付费用户": {
                "priority": "最高",
                "push_channel": ["push", "短信", "站内信"],
                "coupon_type": "vip_discount",
                "content_personalization": "high",
                "push_frequency": "每周2-3次",
                "budget_per_user": 30,
            },
            "潜力付费用户": {
                "priority": "高",
                "push_channel": ["push", "站内信"],
                "coupon_type": "cash_coupon",
                "content_personalization": "high",
                "push_frequency": "每周2次",
                "budget_per_user": 15,
            },
            "普通活跃用户": {
                "priority": "中",
                "push_channel": ["push"],
                "coupon_type": "points_bonus",
                "content_personalization": "medium",
                "push_frequency": "每周1-2次",
                "budget_per_user": 5,
            },
            "轻度沉睡用户": {
                "priority": "高",
                "push_channel": ["push", "站内信"],
                "coupon_type": "exclusive_content",
                "content_personalization": "high",
                "push_frequency": "前3天每天1次",
                "budget_per_user": 10,
            },
            "深度流失用户": {
                "priority": "中",
                "push_channel": ["push"],
                "coupon_type": "cash_coupon",
                "content_personalization": "medium",
                "push_frequency": "每周1次",
                "budget_per_user": 8,
            },
            "新用户": {
                "priority": "高",
                "push_channel": ["push", "站内信"],
                "coupon_type": "exclusive_content",
                "content_personalization": "medium",
                "push_frequency": "前7天每天1次",
                "budget_per_user": 12,
            },
        }

    def generate_recall_plan(self, user_profile, churn_info, cluster_info):
        cluster_name = cluster_info.get("cluster_name", "普通活跃用户")
        strategy = self.strategies.get(cluster_name, self.strategies["普通活跃用户"])
        churn_reason = churn_info.get("churn_reason", "行为衰减")
        risk_level = churn_info.get("risk_level", "中风险")
        prefer_categories = user_profile.get("prefer_categories", "")
        if isinstance(prefer_categories, str) and prefer_categories:
            prefer_list = [c.strip() for c in prefer_categories.split(",") if c.strip()]
        else:
            prefer_list = random.sample(CONTENT_CATEGORIES, 3)
        top_category = prefer_list[0] if prefer_list else "科技"
        push_content = self._generate_push_content(
            churn_reason=churn_reason,
            category=top_category,
            coupon_type=strategy["coupon_type"],
        )
        best_hour = self._predict_best_hour(user_profile)
        recall_plan = {
            "user_id": user_profile.get("user_id", ""),
            "cluster_name": cluster_name,
            "risk_level": risk_level,
            "churn_reason": churn_reason,
            "strategy_priority": strategy["priority"],
            "push_channels": strategy["push_channel"],
            "push_content": push_content,
            "coupon": COUPON_TYPES[strategy["coupon_type"]],
            "best_push_hour": best_hour,
            "push_frequency": strategy["push_frequency"],
            "content_categories": prefer_list[:5],
            "estimated_budget": strategy["budget_per_user"],
            "personalization_level": strategy["content_personalization"],
            "generate_time": datetime.now().isoformat(),
        }
        return recall_plan

    def _generate_push_content(self, churn_reason, category, coupon_type):
        contents = []
        if churn_reason in ["内容不合预期", "互动减少"]:
            templates = PUSH_TEMPLATES["content"]
            for t in templates:
                contents.append(t.format(category=category))
        if coupon_type != "no_coupon":
            coupon = COUPON_TYPES[coupon_type]
            templates = PUSH_TEMPLATES["coupon"]
            for t in templates:
                contents.append(t.format(coupon_value=coupon["value"]))
        templates = PUSH_TEMPLATES["comeback"]
        contents.extend(templates)
        return {
            "primary": contents[0] if contents else "欢迎回来！",
            "alternatives": contents[1:3] if len(contents) > 1 else [],
            "category": category,
        }

    def _predict_best_hour(self, user_profile):
        active_hour_mean = user_profile.get("active_hour_mean", 18)
        if 6 <= active_hour_mean < 10:
            return "早间活跃"
        elif 10 <= active_hour_mean < 14:
            return "午间活跃"
        elif 17 <= active_hour_mean < 22:
            return "晚间活跃"
        elif 22 <= active_hour_mean or active_hour_mean < 2:
            return "夜间活跃"
        else:
            return "通用时段"

    def batch_generate_recalls(self, user_features_df, top_n=None, risk_filter=None):
        plans = []
        df = user_features_df.copy()
        if risk_filter:
            df = df[df["risk_level"].isin(risk_filter)]
        df["recall_priority_score"] = df.apply(self._calc_priority_score, axis=1)
        df = df.sort_values("recall_priority_score", ascending=False)
        if top_n:
            df = df.head(top_n)
        for _, row in df.iterrows():
            user_profile = row.to_dict()
            churn_info = {
                "churn_prob": row.get("churn_prob_30d", 0),
                "risk_level": row.get("risk_level", "中风险"),
                "churn_reason": row.get("churn_reason", "行为衰减"),
            }
            cluster_info = {
                "cluster_id": row.get("cluster_id", 2),
                "cluster_name": row.get("cluster_name", "普通活跃用户"),
            }
            plan = self.generate_recall_plan(user_profile, churn_info, cluster_info)
            plans.append(plan)
        return plans

    def _calc_priority_score(self, row):
        score = 0
        risk_map = {"低风险": 1, "中风险": 2, "高风险": 3, "极高风险": 4}
        score += risk_map.get(row.get("risk_level", "中风险"), 2) * 10
        pay = row.get("total_pay_amount", 0)
        score += min(pay / 50, 10)
        if row.get("is_vip", False):
            score += 15
        active_days = row.get("active_days_window", 0)
        score += (7 - active_days) * 2
        return score


def get_engine():
    return RecallStrategyEngine()


if __name__ == "__main__":
    engine = RecallStrategyEngine()
    print("召回策略引擎初始化完成")
    print(f"已定义 {len(engine.strategies)} 种用户群体策略")
    for name, strat in engine.strategies.items():
        print(f"  - {name}: {strat['priority']}优先级")
