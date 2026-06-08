import os
import sys
import json
import random
import uuid
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RECALL_EFFECTS_FILE = os.path.join(DATA_DIR, "recall_effects.json")
ATTRIBUTION_REPORT_FILE = os.path.join(DATA_DIR, "attribution_reports.json")

PUSH_CHANNELS = ["push", "短信", "站内信", "邮件"]
CONTENT_TYPES = ["纯内容", "内容+优惠券", "纯优惠券", "回归问候"]
RECHURN_WINDOW_DAYS = 7

ATTRIBUTION_DIMENSIONS = [
    "risk_level",
    "cluster_name",
    "push_channel",
    "content_type",
    "churn_reason",
]

METRICS_LIST = [
    "recall_count",
    "revisit_rate",
    "avg_stay_duration",
    "re_interaction_rate",
    "pay_conversion_rate",
    "re_churn_rate",
]


class RecallEffectRecord:
    def __init__(self, user_id, push_channel, content_type, push_time=None):
        self.record_id = str(uuid.uuid4())[:12]
        self.user_id = user_id
        self.push_channel = push_channel
        self.content_type = content_type
        self.push_time = push_time or datetime.now().isoformat()
        self.risk_level = "中风险"
        self.cluster_name = "普通活跃用户"
        self.churn_reason = "行为衰减"
        self.is_revisited = False
        self.revisit_time = None
        self.stay_duration = 0
        self.has_re_interaction = False
        self.interaction_count = 0
        self.has_paid = False
        self.pay_amount = 0
        self.is_re_churned = False
        self.re_churn_days = 0

    def to_dict(self):
        return {
            "record_id": self.record_id,
            "user_id": self.user_id,
            "push_channel": self.push_channel,
            "content_type": self.content_type,
            "push_time": self.push_time,
            "risk_level": self.risk_level,
            "cluster_name": self.cluster_name,
            "churn_reason": self.churn_reason,
            "is_revisited": self.is_revisited,
            "revisit_time": self.revisit_time,
            "stay_duration": self.stay_duration,
            "has_re_interaction": self.has_re_interaction,
            "interaction_count": self.interaction_count,
            "has_paid": self.has_paid,
            "pay_amount": self.pay_amount,
            "is_re_churned": self.is_re_churned,
            "re_churn_days": self.re_churn_days,
        }

    @classmethod
    def from_dict(cls, data):
        record = cls(data["user_id"], data["push_channel"], data["content_type"], data.get("push_time"))
        record.record_id = data["record_id"]
        record.risk_level = data.get("risk_level", "中风险")
        record.cluster_name = data.get("cluster_name", "普通活跃用户")
        record.churn_reason = data.get("churn_reason", "行为衰减")
        record.is_revisited = data.get("is_revisited", False)
        record.revisit_time = data.get("revisit_time")
        record.stay_duration = data.get("stay_duration", 0)
        record.has_re_interaction = data.get("has_re_interaction", False)
        record.interaction_count = data.get("interaction_count", 0)
        record.has_paid = data.get("has_paid", False)
        record.pay_amount = data.get("pay_amount", 0)
        record.is_re_churned = data.get("is_re_churned", False)
        record.re_churn_days = data.get("re_churn_days", 0)
        return record


class RecallEffectsDataStore:
    def __init__(self):
        self.records = {}
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(RECALL_EFFECTS_FILE):
            self._load()

    def _load(self):
        with open(RECALL_EFFECTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            record = RecallEffectRecord.from_dict(item)
            self.records[record.record_id] = record

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        data = [r.to_dict() for r in self.records.values()]
        with open(RECALL_EFFECTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_record(self, record):
        self.records[record.record_id] = record
        self._save()
        return record

    def add_batch_records(self, records):
        for r in records:
            self.records[r.record_id] = r
        self._save()
        return records

    def get_all_records(self):
        return list(self.records.values())

    def get_records_by_user(self, user_id):
        return [r for r in self.records.values() if r.user_id == user_id]

    def get_dataframe(self):
        records = [r.to_dict() for r in self.records.values()]
        return pd.DataFrame(records) if records else pd.DataFrame()

    def generate_mock_data(self, user_features_df, n_records=500):
        records = []
        df = user_features_df.copy()
        if len(df) == 0:
            return records

        for i in range(n_records):
            user_row = df.sample(n=1).iloc[0]
            user_id = user_row["user_id"]
            risk_level = user_row.get("risk_level", "中风险")
            cluster_name = user_row.get("cluster_name", "普通活跃用户")
            churn_reason = user_row.get("churn_reason", "行为衰减")

            push_channel = random.choice(PUSH_CHANNELS)
            content_type = random.choice(CONTENT_TYPES)

            push_time = (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat()

            record = RecallEffectRecord(user_id, push_channel, content_type, push_time)
            record.risk_level = risk_level
            record.cluster_name = cluster_name
            record.churn_reason = churn_reason

            base_revisit_prob = self._get_base_revisit_prob(risk_level, cluster_name)
            channel_bonus = self._get_channel_effect(push_channel)
            content_bonus = self._get_content_effect(content_type, churn_reason)

            revisit_prob = min(0.95, max(0.05, base_revisit_prob * (1 + channel_bonus + content_bonus)))
            record.is_revisited = random.random() < revisit_prob

            if record.is_revisited:
                record.revisit_time = (datetime.fromisoformat(push_time) + timedelta(hours=random.randint(1, 72))).isoformat()
                record.stay_duration = int(np.random.lognormal(5, 0.8))

                interact_prob = min(0.9, max(0.1, revisit_prob * 0.8))
                record.has_re_interaction = random.random() < interact_prob
                if record.has_re_interaction:
                    record.interaction_count = random.randint(1, 20)

                pay_prob = min(0.5, max(0.02, revisit_prob * 0.3))
                if cluster_name in ["高价值活跃用户", "潜力付费用户"]:
                    pay_prob *= 1.5
                record.has_paid = random.random() < pay_prob
                if record.has_paid:
                    record.pay_amount = round(np.random.lognormal(4, 0.7), 2)

                re_churn_prob = self._get_rechurn_prob(risk_level, content_type, stay_duration=record.stay_duration)
                record.is_re_churned = random.random() < re_churn_prob
                if record.is_re_churned:
                    record.re_churn_days = random.randint(1, RECHURN_WINDOW_DAYS)

            records.append(record)
            self.records[record.record_id] = record

        self._save()
        return records

    def _get_base_revisit_prob(self, risk_level, cluster_name):
        risk_map = {
            "低风险": 0.65,
            "中风险": 0.45,
            "高风险": 0.25,
            "极高风险": 0.15,
        }
        cluster_map = {
            "高价值活跃用户": 1.3,
            "潜力付费用户": 1.2,
            "普通活跃用户": 1.0,
            "轻度沉睡用户": 0.9,
            "深度流失用户": 0.6,
            "新用户": 1.1,
        }
        base = risk_map.get(risk_level, 0.4)
        factor = cluster_map.get(cluster_name, 1.0)
        return base * factor

    def _get_channel_effect(self, channel):
        effects = {
            "push": 0.1,
            "短信": 0.15,
            "站内信": 0.2,
            "邮件": -0.05,
        }
        return effects.get(channel, 0)

    def _get_content_effect(self, content_type, churn_reason):
        base_effects = {
            "纯内容": 0.05,
            "内容+优惠券": 0.15,
            "纯优惠券": 0.1,
            "回归问候": 0.0,
        }
        base = base_effects.get(content_type, 0)
        if churn_reason == "内容不合预期" and content_type == "纯内容":
            base += 0.1
        elif churn_reason == "付费意愿低" and "优惠券" in content_type:
            base += 0.15
        elif churn_reason == "活跃度下降" and content_type == "回归问候":
            base += 0.05
        return base

    def _get_rechurn_prob(self, risk_level, content_type, stay_duration=0):
        base = {
            "低风险": 0.1,
            "中风险": 0.2,
            "高风险": 0.35,
            "极高风险": 0.5,
        }.get(risk_level, 0.3)

        content_factor = {
            "纯内容": 1.1,
            "内容+优惠券": 0.8,
            "纯优惠券": 1.0,
            "回归问候": 1.2,
        }.get(content_type, 1.0)

        duration_factor = max(0.5, min(1.5, 1 - stay_duration / 600))

        return min(0.8, base * content_factor * duration_factor)


class AttributionAnalyzer:
    def __init__(self):
        self.data_store = RecallEffectsDataStore()

    def get_overall_stats(self):
        df = self.data_store.get_dataframe()
        if df.empty:
            return self._empty_stats()

        total = len(df)
        revisited = df["is_revisited"].sum()
        revisit_rate = revisited / total if total > 0 else 0

        re_interacted = df[df["is_revisited"]]["has_re_interaction"].sum()
        re_interaction_rate = re_interacted / revisited if revisited > 0 else 0

        paid = df[df["is_revisited"]]["has_paid"].sum()
        pay_conversion_rate = paid / revisited if revisited > 0 else 0

        avg_stay = df[df["is_revisited"]]["stay_duration"].mean() if revisited > 0 else 0

        re_churned = df[df["is_revisited"]]["is_re_churned"].sum()
        re_churn_rate = re_churned / revisited if revisited > 0 else 0

        total_pay = df[df["has_paid"]]["pay_amount"].sum()

        return {
            "total_recalls": total,
            "revisit_count": int(revisited),
            "revisit_rate": round(revisit_rate * 100, 2),
            "re_interaction_count": int(re_interacted),
            "re_interaction_rate": round(re_interaction_rate * 100, 2),
            "pay_conversion_count": int(paid),
            "pay_conversion_rate": round(pay_conversion_rate * 100, 2),
            "avg_stay_duration": round(avg_stay, 1),
            "re_churn_count": int(re_churned),
            "re_churn_rate": round(re_churn_rate * 100, 2),
            "total_pay_amount": round(total_pay, 2),
            "avg_pay_amount": round(total_pay / paid if paid > 0 else 0, 2),
        }

    def analyze_by_dimension(self, dimension):
        df = self.data_store.get_dataframe()
        if df.empty or dimension not in df.columns:
            return []

        results = []
        groups = df.groupby(dimension)

        for group_name, group_df in groups:
            total = len(group_df)
            if total < 5:
                continue

            revisited = group_df["is_revisited"].sum()
            revisit_rate = revisited / total if total > 0 else 0

            re_interacted = group_df[group_df["is_revisited"]]["has_re_interaction"].sum()
            re_interaction_rate = re_interacted / revisited if revisited > 0 else 0

            paid = group_df[group_df["is_revisited"]]["has_paid"].sum()
            pay_conversion_rate = paid / revisited if revisited > 0 else 0

            avg_stay = group_df[group_df["is_revisited"]]["stay_duration"].mean() if revisited > 0 else 0

            re_churned = group_df[group_df["is_revisited"]]["is_re_churned"].sum()
            re_churn_rate = re_churned / revisited if revisited > 0 else 0

            total_pay = group_df[group_df["has_paid"]]["pay_amount"].sum()

            results.append({
                "group": str(group_name),
                "total_recalls": total,
                "revisit_count": int(revisited),
                "revisit_rate": round(revisit_rate * 100, 2),
                "re_interaction_count": int(re_interacted),
                "re_interaction_rate": round(re_interaction_rate * 100, 2),
                "pay_conversion_count": int(paid),
                "pay_conversion_rate": round(pay_conversion_rate * 100, 2),
                "avg_stay_duration": round(avg_stay, 1),
                "re_churn_count": int(re_churned),
                "re_churn_rate": round(re_churn_rate * 100, 2),
                "total_pay_amount": round(total_pay, 2),
            })

        results.sort(key=lambda x: x["revisit_rate"], reverse=True)
        return results

    def analyze_multi_dimension(self, dim1, dim2):
        df = self.data_store.get_dataframe()
        if df.empty or dim1 not in df.columns or dim2 not in df.columns:
            return {}

        result = {}
        groups = df.groupby([dim1, dim2])

        for (g1, g2), group_df in groups:
            total = len(group_df)
            if total < 3:
                continue
            revisited = group_df["is_revisited"].sum()
            revisit_rate = revisited / total if total > 0 else 0

            if g1 not in result:
                result[g1] = {}
            result[g1][g2] = {
                "total_recalls": total,
                "revisit_rate": round(revisit_rate * 100, 2),
                "revisit_count": int(revisited),
            }

        return result

    def analyze_rechurn_reasons(self):
        df = self.data_store.get_dataframe()
        if df.empty:
            return []

        re_churned_df = df[(df["is_revisited"] == True) & (df["is_re_churned"] == True)]

        if re_churned_df.empty:
            return []

        reasons = []

        content_mismatch_count = 0
        for _, row in re_churned_df.iterrows():
            if row["content_type"] == "纯内容" and row["stay_duration"] < 60:
                content_mismatch_count += 1
            elif row["churn_reason"] == "内容不合预期" and row["content_type"] == "纯内容":
                content_mismatch_count += 1

        total_rechurn = len(re_churned_df)

        welfare_insufficient_count = 0
        for _, row in re_churned_df.iterrows():
            if "优惠券" in row["content_type"] and row["pay_amount"] == 0 and row["stay_duration"] < 120:
                welfare_insufficient_count += 1
            elif row["churn_reason"] == "付费意愿低" and not row["has_paid"]:
                welfare_insufficient_count += 1

        experience_issue_count = 0
        for _, row in re_churned_df.iterrows():
            if row["stay_duration"] < 30 and row["interaction_count"] == 0:
                experience_issue_count += 1
            elif row["push_channel"] == "邮件" and not row["has_re_interaction"]:
                experience_issue_count += 0.5

        reasons.append({
            "reason": "内容不匹配",
            "count": int(content_mismatch_count),
            "ratio": round(content_mismatch_count / total_rechurn * 100, 2) if total_rechurn > 0 else 0,
            "description": "推送内容与用户兴趣不符，用户快速流失",
            "related_factors": ["content_type", "churn_reason", "stay_duration"],
        })

        reasons.append({
            "reason": "福利吸引力不足",
            "count": int(welfare_insufficient_count),
            "ratio": round(welfare_insufficient_count / total_rechurn * 100, 2) if total_rechurn > 0 else 0,
            "description": "优惠券/福利力度不够，未能有效留存用户",
            "related_factors": ["content_type", "pay_amount", "churn_reason"],
        })

        reasons.append({
            "reason": "体验问题",
            "count": int(experience_issue_count),
            "ratio": round(experience_issue_count / total_rechurn * 100, 2) if total_rechurn > 0 else 0,
            "description": "产品体验不佳或触达渠道效果差，用户再次流失",
            "related_factors": ["push_channel", "stay_duration", "interaction_count"],
        })

        other_count = total_rechurn - content_mismatch_count - welfare_insufficient_count - experience_issue_count
        if other_count > 0:
            reasons.append({
                "reason": "其他原因",
                "count": int(other_count),
                "ratio": round(other_count / total_rechurn * 100, 2) if total_rechurn > 0 else 0,
                "description": "其他不确定因素导致的再次流失",
                "related_factors": [],
            })

        reasons.sort(key=lambda x: x["count"], reverse=True)
        return reasons

    def generate_optimization_report(self):
        overall = self.get_overall_stats()
        if overall["total_recalls"] == 0:
            return {
                "report_time": datetime.now().isoformat(),
                "status": "no_data",
                "message": "暂无召回效果数据，无法生成优化报告",
                "overall_stats": overall,
                "dimension_analysis": {},
                "rechurn_attribution": [],
                "optimization_suggestions": [],
                "summary": {
                    "total_suggestions": 0,
                    "high_priority_count": 0,
                    "medium_priority_count": 0,
                },
            }

        risk_analysis = self.analyze_by_dimension("risk_level")
        cluster_analysis = self.analyze_by_dimension("cluster_name")
        channel_analysis = self.analyze_by_dimension("push_channel")
        content_analysis = self.analyze_by_dimension("content_type")
        rechurn_reasons = self.analyze_rechurn_reasons()

        suggestions = []

        if channel_analysis:
            best_channel = max(channel_analysis, key=lambda x: x["revisit_rate"])
            worst_channel = min(channel_analysis, key=lambda x: x["revisit_rate"])
            suggestions.append({
                "category": "渠道优化",
                "priority": "高" if (best_channel["revisit_rate"] - worst_channel["revisit_rate"]) > 10 else "中",
                "suggestion": f"重点投入效果最好的「{best_channel['group']}」渠道（复访率{best_channel['revisit_rate']}%），优化或减少「{worst_channel['group']}」渠道的投入",
                "expected_impact": f"预计可提升整体复访率{(best_channel['revisit_rate'] - worst_channel['revisit_rate']) * 0.3:.1f}%",
            })

        if content_analysis:
            best_content = max(content_analysis, key=lambda x: x["revisit_rate"])
            worst_content = min(content_analysis, key=lambda x: x["revisit_rate"])
            suggestions.append({
                "category": "内容策略优化",
                "priority": "高",
                "suggestion": f"主推「{best_content['group']}」类型内容（复访率{best_content['revisit_rate']}%），改进「{worst_content['group']}」内容质量或减少推送",
                "expected_impact": f"预计可提升内容匹配度，减少内容不匹配导致的流失",
            })

        if rechurn_reasons and rechurn_reasons[0]["reason"] == "内容不匹配":
            suggestions.append({
                "category": "内容匹配提升",
                "priority": "高",
                "suggestion": "优化个性化推荐算法，提升内容与用户兴趣的匹配度，增加内容多样性测试",
                "expected_impact": f"预计可降低{rechurn_reasons[0]['ratio'] * 0.2:.1f}%的再流失率",
            })

        if rechurn_reasons and any(r["reason"] == "福利吸引力不足" for r in rechurn_reasons):
            welfare_reason = next(r for r in rechurn_reasons if r["reason"] == "福利吸引力不足")
            suggestions.append({
                "category": "福利策略优化",
                "priority": "中",
                "suggestion": "针对价格敏感型用户，加大优惠券力度或设计阶梯式福利，提升用户付费转化和留存",
                "expected_impact": f"预计可提升付费转化率{welfare_reason['ratio'] * 0.15:.1f}%",
            })

        if risk_analysis:
            high_risk_data = [r for r in risk_analysis if "高" in r["group"]]
            if high_risk_data:
                suggestions.append({
                    "category": "高风险用户策略",
                    "priority": "中",
                    "suggestion": "针对高风险和极高风险用户，采用更激进的召回策略，组合多种渠道和福利",
                    "expected_impact": f"预计可提升高风险用户召回率5-8%",
                })

        if cluster_analysis:
            best_cluster = max(cluster_analysis, key=lambda x: x["pay_conversion_rate"])
            suggestions.append({
                "category": "高价值群体深耕",
                "priority": "中",
                "suggestion": f"重点维护「{best_cluster['group']}」群体，提供专属内容和VIP服务，提升LTV",
                "expected_impact": f"该群体付费转化率已达{best_cluster['pay_conversion_rate']}%，持续运营可提升ARPU",
            })

        top_rechurn = rechurn_reasons[0] if rechurn_reasons else None
        if top_rechurn and top_rechurn["ratio"] > 30:
            suggestions.append({
                "category": "再流失治理",
                "priority": "高" if top_rechurn["ratio"] > 40 else "中",
                "suggestion": f"首要解决「{top_rechurn['reason']}」问题（占再流失的{top_rechurn['ratio']}%），制定专项优化方案",
                "expected_impact": f"针对性优化后预计可降低再流失率{top_rechurn['ratio'] * 0.25:.1f}%",
            })

        suggestions.sort(key=lambda x: {"高": 0, "中": 1, "低": 2}[x["priority"]])

        return {
            "report_time": datetime.now().isoformat(),
            "overall_stats": overall,
            "dimension_analysis": {
                "by_risk_level": risk_analysis,
                "by_cluster": cluster_analysis,
                "by_channel": channel_analysis,
                "by_content_type": content_analysis,
            },
            "rechurn_attribution": rechurn_reasons,
            "optimization_suggestions": suggestions,
            "summary": {
                "total_suggestions": len(suggestions),
                "high_priority_count": sum(1 for s in suggestions if s["priority"] == "高"),
                "medium_priority_count": sum(1 for s in suggestions if s["priority"] == "中"),
            },
        }

    def _empty_stats(self):
        return {
            "total_recalls": 0,
            "revisit_count": 0,
            "revisit_rate": 0,
            "re_interaction_count": 0,
            "re_interaction_rate": 0,
            "pay_conversion_count": 0,
            "pay_conversion_rate": 0,
            "avg_stay_duration": 0,
            "re_churn_count": 0,
            "re_churn_rate": 0,
            "total_pay_amount": 0,
            "avg_pay_amount": 0,
        }

    def get_trend_data(self, days=14):
        df = self.data_store.get_dataframe()
        if df.empty:
            return []

        df["push_date"] = pd.to_datetime(df["push_time"]).dt.date

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days - 1)

        date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        trend_data = []

        for d in date_range:
            date_str = d.strftime("%Y-%m-%d")
            day_df = df[df["push_date"] == d.date()]
            total = len(day_df)
            if total == 0:
                trend_data.append({
                    "date": date_str,
                    "total_recalls": 0,
                    "revisit_rate": 0,
                    "pay_conversion_rate": 0,
                    "re_churn_rate": 0,
                })
                continue

            revisited = day_df["is_revisited"].sum()
            revisit_rate = revisited / total if total > 0 else 0

            paid = day_df[day_df["is_revisited"]]["has_paid"].sum()
            pay_rate = paid / revisited if revisited > 0 else 0

            re_churned = day_df[day_df["is_revisited"]]["is_re_churned"].sum()
            rechurn_rate = re_churned / revisited if revisited > 0 else 0

            trend_data.append({
                "date": date_str,
                "total_recalls": total,
                "revisit_rate": round(revisit_rate * 100, 2),
                "pay_conversion_rate": round(pay_rate * 100, 2),
                "re_churn_rate": round(rechurn_rate * 100, 2),
            })

        return trend_data


def get_attribution_analyzer():
    return AttributionAnalyzer()


if __name__ == "__main__":
    analyzer = AttributionAnalyzer()
    print("召回效果归因分析模块初始化完成")

    overall = analyzer.get_overall_stats()
    print(f"\n整体效果统计:")
    print(f"  总召回数: {overall['total_recalls']}")
    print(f"  复访率: {overall['revisit_rate']}%")
    print(f"  付费转化率: {overall['pay_conversion_rate']}%")
    print(f"  再流失率: {overall['re_churn_rate']}%")

    print(f"\n按渠道分析:")
    channel_stats = analyzer.analyze_by_dimension("push_channel")
    for s in channel_stats:
        print(f"  {s['group']}: 复访率{s['revisit_rate']}%, 付费转化{s['pay_conversion_rate']}%")

    print(f"\n再流失归因:")
    reasons = analyzer.analyze_rechurn_reasons()
    for r in reasons:
        print(f"  {r['reason']}: {r['count']}人 ({r['ratio']}%)")

    report = analyzer.generate_optimization_report()
    print(f"\n优化建议 ({report['summary']['total_suggestions']}条):")
    for s in report["optimization_suggestions"]:
        print(f"  [{s['priority']}] {s['category']}: {s['suggestion']}")
