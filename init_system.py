#!/usr/bin/env python3
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

def init_all():
    print("=" * 60)
    print("  AI智能用户召回系统 - 一键初始化")
    print("=" * 60)
    print()

    print("1. 生成模拟用户数据...")
    from src.data.generator import generate_all_data
    users, behaviors, labels = generate_all_data(n_users=2000, days=60)
    print(f"   ✓ 生成完成：2000用户，60天行为数据")
    print()

    print("2. 构建用户特征...")
    from src.data.features import load_raw_data, build_user_features
    users, behaviors, labels = load_raw_data()
    features = build_user_features(users, behaviors, labels)
    features.to_csv(os.path.join(BASE_DIR, "data/user_features.csv"), index=False)
    print(f"   ✓ 特征维度：{len(features.columns)-1}维，样本数：{len(features)}")
    print()

    print("3. 训练流失预测模型...")
    from src.data.features import get_feature_matrix
    from src.models.churn_predictor import ChurnPredictor
    X, _ = get_feature_matrix(features)
    y = features["is_churned_30d"].values
    predictor = ChurnPredictor(model_type="gb")
    metrics = predictor.train(X, y)
    predictor.save(os.path.join(BASE_DIR, "models/churn_model.pkl"))
    print(f"   ✓ 模型准确率: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['roc_auc']:.4f}")
    print()

    print("4. 训练用户聚类模型...")
    from src.data.features import FEATURE_COLUMNS
    from src.models.user_clustering import UserClustering, CLUSTER_NAMES
    clustering = UserClustering(n_clusters=6)
    cluster_labels, silhouette = clustering.fit(X, FEATURE_COLUMNS)
    clustering.save(os.path.join(BASE_DIR, "models/clustering_model.pkl"))
    print(f"   ✓ 轮廓系数: {silhouette:.4f}, 6个用户群体")
    for cid, name in sorted(CLUSTER_NAMES.items()):
        size = (cluster_labels == cid).sum()
        print(f"     - {name}: {size}人")
    print()

    print("5. 初始化A/B实验...")
    from src.models.ab_experiment import create_default_experiments
    manager = create_default_experiments()
    print(f"   ✓ 创建 {len(manager.list_experiments())} 个默认实验")
    print()

    print("=" * 60)
    print("  ✅ 系统初始化完成！")
    print("=" * 60)
    print()
    print("启动命令:  python run.py")
    print("访问地址:  http://localhost:8000")
    print()

if __name__ == "__main__":
    init_all()
