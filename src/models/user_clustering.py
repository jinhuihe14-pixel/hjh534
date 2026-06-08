import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.data.features import load_raw_data, build_user_features, get_feature_matrix, FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")


CLUSTER_NAMES = {
    0: "高价值活跃用户",
    1: "潜力付费用户",
    2: "普通活跃用户",
    3: "轻度沉睡用户",
    4: "深度流失用户",
    5: "新用户",
}


class UserClustering:
    def __init__(self, n_clusters=6):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.pca = None
        self.cluster_profiles_ = {}
        self.feature_names = FEATURE_COLUMNS

    def fit(self, X, feature_names=None):
        if feature_names is not None:
            self.feature_names = feature_names
        X_scaled = self.scaler.fit_transform(X)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X_scaled)
        self.pca = PCA(n_components=2, random_state=42)
        self.pca.fit(X_scaled)
        self._build_cluster_profiles(X, labels)
        score = silhouette_score(X_scaled, labels)
        return labels, score

    def _build_cluster_profiles(self, X, labels):
        df = pd.DataFrame(X, columns=self.feature_names)
        df["cluster"] = labels
        overall_mean = df[self.feature_names].mean()
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df["cluster"] == cluster_id]
            cluster_mean = cluster_data[self.feature_names].mean()
            relative = (cluster_mean - overall_mean) / (overall_mean + 1e-8)
            top_features = relative.abs().sort_values(ascending=False).head(10)
            self.cluster_profiles_[cluster_id] = {
                "cluster_id": cluster_id,
                "size": len(cluster_data),
                "size_ratio": len(cluster_data) / len(df),
                "top_features": top_features.to_dict(),
                "mean_values": cluster_mean.to_dict(),
                "relative_values": relative.to_dict(),
            }

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.kmeans.predict(X_scaled)

    def get_cluster_name(self, cluster_id):
        return CLUSTER_NAMES.get(int(cluster_id), f"群体{cluster_id}")

    def get_cluster_profile(self, cluster_id):
        return self.cluster_profiles_.get(int(cluster_id), {})

    def save(self, path):
        joblib.dump({
            "kmeans": self.kmeans,
            "scaler": self.scaler,
            "pca": self.pca,
            "cluster_profiles": self.cluster_profiles_,
            "feature_names": self.feature_names,
            "n_clusters": self.n_clusters,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        clustering = cls(n_clusters=data["n_clusters"])
        clustering.kmeans = data["kmeans"]
        clustering.scaler = data["scaler"]
        clustering.pca = data["pca"]
        clustering.cluster_profiles_ = data["cluster_profiles"]
        clustering.feature_names = data["feature_names"]
        return clustering


def train_clustering():
    users, behaviors, labels = load_raw_data()
    features = build_user_features(users, behaviors, labels)
    X, feature_names = get_feature_matrix(features)
    clustering = UserClustering(n_clusters=6)
    cluster_labels, score = clustering.fit(X, feature_names)
    print(f"用户分群完成！轮廓系数: {score:.4f}")
    features["cluster_id"] = cluster_labels
    features["cluster_name"] = features["cluster_id"].map(CLUSTER_NAMES)
    print("\n群体分布:")
    for cid in sorted(CLUSTER_NAMES.keys()):
        profile = clustering.get_cluster_profile(cid)
        name = CLUSTER_NAMES[cid]
        size = profile.get("size", 0)
        ratio = profile.get("size_ratio", 0)
        print(f"  {name} (ID:{cid}): {size}人 ({ratio*100:.1f}%)")
    model_path = os.path.join(MODEL_DIR, "clustering_model.pkl")
    clustering.save(model_path)
    print(f"\n聚类模型已保存: {model_path}")
    features.to_csv(os.path.join(DATA_DIR, "user_features.csv"), index=False)
    return clustering, features


if __name__ == "__main__":
    train_clustering()
