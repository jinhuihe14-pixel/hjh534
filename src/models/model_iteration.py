import os
import sys
import json
import uuid
from datetime import datetime
from collections import defaultdict

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.features import FEATURE_COLUMNS, get_feature_matrix
from src.models.churn_predictor import ChurnPredictor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.json")
WHITELIST_FILE = os.path.join(DATA_DIR, "strategy_whitelist.json")
MODEL_UPDATE_LOG_FILE = os.path.join(DATA_DIR, "model_update_log.json")

ANNOTATION_TYPES = [
    "churn_prediction_error",
    "recall_effect_poor",
    "churn_reason_error",
    "cluster_error",
    "other",
]

ANNOTATION_LABELS = {
    "churn_prediction_error": ["false_positive", "false_negative"],
    "recall_effect_poor": ["content_mismatch", "coupon_useless", "channel_bad", "other"],
    "churn_reason_error": ["wrong_reason", "missing_reason"],
    "cluster_error": ["wrong_cluster", "should_be_new_cluster"],
    "other": [],
}


class AnnotationRecord:
    def __init__(self, user_id, annotation_type, label, note="", operator="system"):
        self.annotation_id = str(uuid.uuid4())[:12]
        self.user_id = user_id
        self.annotation_type = annotation_type
        self.label = label
        self.note = note
        self.operator = operator
        self.create_time = datetime.now().isoformat()
        self.is_used_for_training = False
        self.use_time = None

    def to_dict(self):
        return {
            "annotation_id": self.annotation_id,
            "user_id": self.user_id,
            "annotation_type": self.annotation_type,
            "label": self.label,
            "note": self.note,
            "operator": self.operator,
            "create_time": self.create_time,
            "is_used_for_training": self.is_used_for_training,
            "use_time": self.use_time,
        }

    @classmethod
    def from_dict(cls, data):
        record = cls(
            data["user_id"],
            data["annotation_type"],
            data["label"],
            data.get("note", ""),
            data.get("operator", "system"),
        )
        record.annotation_id = data["annotation_id"]
        record.create_time = data.get("create_time", record.create_time)
        record.is_used_for_training = data.get("is_used_for_training", False)
        record.use_time = data.get("use_time")
        return record


class AnnotationManager:
    def __init__(self):
        self.annotations = {}
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(ANNOTATIONS_FILE):
            self._load()

    def _load(self):
        with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            ann = AnnotationRecord.from_dict(item)
            self.annotations[ann.annotation_id] = ann

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        data = [a.to_dict() for a in self.annotations.values()]
        with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_annotation(self, user_id, annotation_type, label, note="", operator="system"):
        if annotation_type not in ANNOTATION_TYPES:
            raise ValueError(f"未知标注类型: {annotation_type}")
        valid_labels = ANNOTATION_LABELS.get(annotation_type, [])
        if valid_labels and label not in valid_labels:
            raise ValueError(f"无效标签: {label}，有效标签: {valid_labels}")

        ann = AnnotationRecord(user_id, annotation_type, label, note, operator)
        self.annotations[ann.annotation_id] = ann
        self._save()
        return ann

    def get_annotation(self, annotation_id):
        return self.annotations.get(annotation_id)

    def get_user_annotations(self, user_id):
        return [a for a in self.annotations.values() if a.user_id == user_id]

    def list_annotations(self, annotation_type=None, is_used=None, page=1, page_size=20):
        results = list(self.annotations.values())

        if annotation_type:
            results = [a for a in results if a.annotation_type == annotation_type]
        if is_used is not None:
            results = [a for a in results if a.is_used_for_training == is_used]

        results.sort(key=lambda x: x.create_time, reverse=True)
        total = len(results)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = [a.to_dict() for a in results[start:end]]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "data": page_data,
        }

    def get_unused_annotations(self, annotation_type=None):
        results = [a for a in self.annotations.values() if not a.is_used_for_training]
        if annotation_type:
            results = [a for a in results if a.annotation_type == annotation_type]
        return results

    def mark_as_used(self, annotation_ids):
        count = 0
        for aid in annotation_ids:
            if aid in self.annotations:
                self.annotations[aid].is_used_for_training = True
                self.annotations[aid].use_time = datetime.now().isoformat()
                count += 1
        self._save()
        return count

    def get_stats(self):
        total = len(self.annotations)
        by_type = defaultdict(int)
        by_used = {"used": 0, "unused": 0}

        for ann in self.annotations.values():
            by_type[ann.annotation_type] += 1
            if ann.is_used_for_training:
                by_used["used"] += 1
            else:
                by_used["unused"] += 1

        return {
            "total": total,
            "by_type": dict(by_type),
            "by_used": by_used,
        }

    def delete_annotation(self, annotation_id):
        if annotation_id in self.annotations:
            del self.annotations[annotation_id]
            self._save()
            return True
        return False

    def get_annotation_types(self):
        return [
            {
                "type": t,
                "labels": ANNOTATION_LABELS[t],
                "description": self._get_type_description(t),
            }
            for t in ANNOTATION_TYPES
        ]

    def _get_type_description(self, ann_type):
        descriptions = {
            "churn_prediction_error": "流失预测错误，包括假阳性（预测流失但实际活跃）和假阴性（预测活跃但实际流失）",
            "recall_effect_poor": "召回效果差，包括内容不匹配、优惠券无效、渠道效果差等",
            "churn_reason_error": "流失原因归因错误",
            "cluster_error": "用户分群错误",
            "other": "其他标注类型",
        }
        return descriptions.get(ann_type, "")


class WhitelistRecord:
    def __init__(self, user_id, reason="", operator="system", custom_strategy=None):
        self.user_id = user_id
        self.reason = reason
        self.operator = operator
        self.create_time = datetime.now().isoformat()
        self.custom_strategy = custom_strategy or {}
        self.is_active = True

    def to_dict(self):
        return {
            "user_id": self.user_id,
            "reason": self.reason,
            "operator": self.operator,
            "create_time": self.create_time,
            "custom_strategy": self.custom_strategy,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data):
        record = cls(
            data["user_id"],
            data.get("reason", ""),
            data.get("operator", "system"),
            data.get("custom_strategy", {}),
        )
        record.create_time = data.get("create_time", record.create_time)
        record.is_active = data.get("is_active", True)
        return record


class StrategyWhitelistManager:
    def __init__(self):
        self.whitelist = {}
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(WHITELIST_FILE):
            self._load()

    def _load(self):
        with open(WHITELIST_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            record = WhitelistRecord.from_dict(item)
            self.whitelist[record.user_id] = record

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        data = [r.to_dict() for r in self.whitelist.values()]
        with open(WHITELIST_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_to_whitelist(self, user_id, reason="", operator="system", custom_strategy=None):
        record = WhitelistRecord(user_id, reason, operator, custom_strategy)
        self.whitelist[user_id] = record
        self._save()
        return record

    def remove_from_whitelist(self, user_id):
        if user_id in self.whitelist:
            self.whitelist[user_id].is_active = False
            self._save()
            return True
        return False

    def is_whitelisted(self, user_id):
        record = self.whitelist.get(user_id)
        return record is not None and record.is_active

    def get_whitelist_record(self, user_id):
        record = self.whitelist.get(user_id)
        return record.to_dict() if record and record.is_active else None

    def list_whitelist(self, active_only=True, page=1, page_size=20):
        results = list(self.whitelist.values())
        if active_only:
            results = [r for r in results if r.is_active]

        results.sort(key=lambda x: x.create_time, reverse=True)
        total = len(results)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = [r.to_dict() for r in results[start:end]]

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "data": page_data,
        }

    def update_custom_strategy(self, user_id, custom_strategy):
        if user_id in self.whitelist:
            self.whitelist[user_id].custom_strategy = custom_strategy
            self._save()
            return True
        return False

    def get_stats(self):
        total = len(self.whitelist)
        active = sum(1 for r in self.whitelist.values() if r.is_active)
        inactive = total - active
        return {
            "total": total,
            "active": active,
            "inactive": inactive,
        }


class ModelUpdateLog:
    def __init__(self):
        self.logs = []
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(MODEL_UPDATE_LOG_FILE):
            with open(MODEL_UPDATE_LOG_FILE, "r", encoding="utf-8") as f:
                self.logs = json.load(f)

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(MODEL_UPDATE_LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

    def add_log(self, update_type, metrics=None, annotation_count=0, operator="system", note=""):
        log = {
            "update_id": str(uuid.uuid4())[:12],
            "update_type": update_type,
            "update_time": datetime.now().isoformat(),
            "metrics": metrics or {},
            "annotation_count": annotation_count,
            "operator": operator,
            "note": note,
        }
        self.logs.insert(0, log)
        if len(self.logs) > 100:
            self.logs = self.logs[:100]
        self._save()
        return log

    def list_logs(self, page=1, page_size=20):
        total = len(self.logs)
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "data": self.logs[start:end],
        }

    def get_latest(self):
        return self.logs[0] if self.logs else None


class IncrementalModelUpdater:
    def __init__(self, base_model_path=None):
        self.base_model_path = base_model_path or os.path.join(MODEL_DIR, "churn_model.pkl")
        self.annotation_manager = AnnotationManager()
        self.update_log = ModelUpdateLog()

    def incremental_update(self, features_df, annotations=None):
        if annotations is None:
            annotations = self.annotation_manager.get_unused_annotations("churn_prediction_error")

        if not annotations:
            return {
                "success": False,
                "message": "没有可用的标注数据进行增量更新",
            }

        if not os.path.exists(self.base_model_path):
            return {
                "success": False,
                "message": "基础模型不存在，请先训练初始模型",
            }

        try:
            base_predictor = ChurnPredictor.load(self.base_model_path)
        except Exception as e:
            return {
                "success": False,
                "message": f"加载基础模型失败: {e}",
            }

        label_updates = {}
        for ann in annotations:
            if ann.label == "false_positive":
                label_updates[ann.user_id] = 0
            elif ann.label == "false_negative":
                label_updates[ann.user_id] = 1

        update_count = len(label_updates)
        if update_count == 0:
            return {
                "success": False,
                "message": "没有可用于更新的流失预测标注",
            }

        X_all, feature_names = get_feature_matrix(features_df)
        user_ids = features_df["user_id"].values

        original_predictions = base_predictor.predict_proba(X_all)

        updated_indices = []
        updated_labels = []
        for i, uid in enumerate(user_ids):
            if uid in label_updates:
                updated_indices.append(i)
                updated_labels.append(label_updates[uid])

        if len(updated_indices) < 10:
            return {
                "success": False,
                "message": f"标注样本不足（仅{len(updated_indices)}条），至少需要10条才能进行增量更新",
            }

        all_labels = features_df["is_churned_30d"].values.copy()
        for idx, new_label in zip(updated_indices, updated_labels):
            all_labels[idx] = new_label

        learning_rate = min(0.3, len(updated_indices) / len(all_labels) * 2)

        metrics_before = self._evaluate_model(base_predictor, X_all, all_labels)

        updated_predictor = self._fine_tune_model(
            base_predictor, X_all, all_labels, updated_indices, learning_rate
        )

        metrics_after = self._evaluate_model(updated_predictor, X_all, all_labels)

        backup_path = self.base_model_path + ".backup"
        import shutil
        shutil.copy2(self.base_model_path, backup_path)

        updated_predictor.save(self.base_model_path)

        self.annotation_manager.mark_as_used([a.annotation_id for a in annotations])

        self.update_log.add_log(
            update_type="incremental_update",
            metrics={
                "before": metrics_before,
                "after": metrics_after,
            },
            annotation_count=update_count,
            note=f"增量更新完成，使用{update_count}条标注数据，学习率{learning_rate:.3f}",
        )

        return {
            "success": True,
            "message": "模型增量更新成功",
            "annotation_count": update_count,
            "learning_rate": learning_rate,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "improvement": {
                "accuracy": round(metrics_after["accuracy"] - metrics_before["accuracy"], 4),
                "f1": round(metrics_after["f1"] - metrics_before["f1"], 4),
                "roc_auc": round(metrics_after["roc_auc"] - metrics_before["roc_auc"], 4),
            },
        }

    def _fine_tune_model(self, base_predictor, X, y, update_indices, learning_rate=0.1):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        X_scaled = base_predictor.scaler.transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        n_estimators_original = base_predictor.model.n_estimators
        new_n_estimators = int(n_estimators_original * (1 + learning_rate * 0.5))

        base_predictor.model.set_params(
            n_estimators=new_n_estimators,
            learning_rate=base_predictor.model.learning_rate * learning_rate,
        )

        base_predictor.model.fit(X_train, y_train)

        return base_predictor

    def _evaluate_model(self, predictor, X, y):
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        X_scaled = predictor.scaler.transform(X)
        y_pred = predictor.model.predict(X_scaled)
        y_prob = predictor.model.predict_proba(X_scaled)[:, 1]

        return {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred), 4),
            "recall": round(recall_score(y, y_pred), 4),
            "f1": round(f1_score(y, y_pred), 4),
            "roc_auc": round(roc_auc_score(y, y_prob), 4),
        }

    def get_update_history(self, page=1, page_size=20):
        return self.update_log.list_logs(page, page_size)


def get_annotation_manager():
    return AnnotationManager()


def get_whitelist_manager():
    return StrategyWhitelistManager()


def get_model_updater():
    return IncrementalModelUpdater()


if __name__ == "__main__":
    print("模型迭代辅助模块初始化")

    ann_manager = AnnotationManager()
    stats = ann_manager.get_stats()
    print(f"标注统计: {stats['total']}条记录")
    print(f"  已用于训练: {stats['by_used']['used']}")
    print(f"  未使用: {stats['by_used']['unused']}")

    whitelist_manager = StrategyWhitelistManager()
    wl_stats = whitelist_manager.get_stats()
    print(f"\n白名单统计: {wl_stats['active']}个活跃用户")

    updater = IncrementalModelUpdater()
    history = updater.get_update_history(1, 5)
    print(f"\n模型更新历史: {history['total']}次更新")
