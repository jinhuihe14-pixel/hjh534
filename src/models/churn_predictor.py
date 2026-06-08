import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.features import load_raw_data, build_user_features, get_feature_matrix, FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(MODEL_DIR, exist_ok=True)


class ChurnPredictor:
    def __init__(self, model_type="gb"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = FEATURE_COLUMNS
        self.feature_importance_ = None

    def _build_model(self):
        if self.model_type == "gb":
            return GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.08,
                subsample=0.85, random_state=42
            )
        elif self.model_type == "rf":
            return RandomForestClassifier(
                n_estimators=200, max_depth=8, random_state=42
            )
        elif self.model_type == "lr":
            return LogisticRegression(max_iter=1000)
        else:
            raise ValueError(f"未知模型类型: {self.model_type}")

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.model = self._build_model()
        self.model.fit(X_train_scaled, y_train)
        if hasattr(self.model, "feature_importances_"):
            self.feature_importance_ = dict(zip(
                self.feature_names, self.model.feature_importances_
            ))
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        return metrics

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict_risk_level(self, X, days=30):
        prob = self.predict_proba(X)
        scale_factor = days / 30.0
        adjusted_prob = np.clip(prob * scale_factor, 0, 1)
        levels = []
        for p in adjusted_prob:
            if p < 0.2:
                levels.append("低风险")
            elif p < 0.5:
                levels.append("中风险")
            elif p < 0.8:
                levels.append("高风险")
            else:
                levels.append("极高风险")
        return adjusted_prob, levels

    def attribute_churn_reason(self, X, feature_names=None):
        if feature_names is None:
            feature_names = self.feature_names
        if self.feature_importance_ is None:
            return ["行为衰减"] * len(X)
        X_scaled = self.scaler.transform(X)
        reasons = []
        key_features = {
            "内容不合预期": ["click_through_rate", "like_rate", "total_browse"],
            "互动减少": ["total_click", "total_like", "comment_count", "share_count"],
            "活跃度下降": ["active_days_window", "sessions_7d", "decay_ratio"],
            "付费意愿低": ["total_pay_amount", "is_vip"],
        }
        for idx in range(len(X)):
            scores = {}
            for reason, feats in key_features.items():
                score = 0
                count = 0
                for f in feats:
                    if f in self.feature_importance_:
                        fi = self.feature_importance_[f]
                        col_idx = self.feature_names.index(f) if f in self.feature_names else -1
                        if col_idx >= 0:
                            val = X_scaled[idx][col_idx]
                            score += fi * max(0, -val)
                            count += 1
                scores[reason] = score / max(count, 1)
            top_reason = max(scores, key=scores.get)
            reasons.append(top_reason)
        return reasons

    def save(self, path):
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_importance": self.feature_importance_,
            "feature_names": self.feature_names,
            "model_type": self.model_type,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        predictor = cls(model_type=data["model_type"])
        predictor.model = data["model"]
        predictor.scaler = data["scaler"]
        predictor.feature_importance_ = data["feature_importance"]
        predictor.feature_names = data["feature_names"]
        return predictor


def train_churn_model():
    users, behaviors, labels = load_raw_data()
    features = build_user_features(users, behaviors, labels)
    X, feature_names = get_feature_matrix(features)
    y = features["is_churned_30d"].values
    predictor = ChurnPredictor(model_type="gb")
    metrics = predictor.train(X, y)
    print("模型训练完成！")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  精确率: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1值:   {metrics['f1']:.4f}")
    print(f"  AUC:    {metrics['roc_auc']:.4f}")
    model_path = os.path.join(MODEL_DIR, "churn_model.pkl")
    predictor.save(model_path)
    print(f"模型已保存: {model_path}")
    prob, levels = predictor.predict_risk_level(X)
    print(f"风险等级分布: {pd.Series(levels).value_counts().to_dict()}")
    reasons = predictor.attribute_churn_reason(X)
    print(f"流失归因分布: {pd.Series(reasons).value_counts().to_dict()}")
    return predictor, metrics


if __name__ == "__main__":
    train_churn_model()
