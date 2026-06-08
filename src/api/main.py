import os
import sys
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.generator import generate_all_data
from src.data.features import load_raw_data, build_user_features, get_feature_matrix, FEATURE_COLUMNS
from src.models.churn_predictor import ChurnPredictor
from src.models.user_clustering import UserClustering, CLUSTER_NAMES
from src.models.recall_engine import RecallStrategyEngine
from src.models.ab_experiment import ABExperimentManager, create_default_experiments

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
STATIC_DIR = os.path.join(BASE_DIR, "frontend")

app = FastAPI(
    title="AI智能用户召回系统",
    description="用户画像、流失预测、智能分群、个性化召回一体化AI系统",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SystemState:
    def __init__(self):
        self.churn_model = None
        self.clustering_model = None
        self.recall_engine = None
        self.ab_manager = None
        self.features_df = None
        self.initialized = False

state = SystemState()


def ensure_initialized():
    if not state.initialized:
        raise HTTPException(status_code=500, detail="系统尚未初始化，请先调用 /api/system/init")


@app.on_event("startup")
async def startup_event():
    print("正在启动服务...")
    if os.path.exists(os.path.join(MODEL_DIR, "churn_model.pkl")):
        try:
            load_system()
            print("系统模型已加载")
        except Exception as e:
            print(f"加载模型失败: {e}")


def load_system():
    state.churn_model = ChurnPredictor.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
    state.clustering_model = UserClustering.load(os.path.join(MODEL_DIR, "clustering_model.pkl"))
    state.recall_engine = RecallStrategyEngine()
    state.ab_manager = create_default_experiments()
    users, behaviors, labels = load_raw_data()
    state.features_df = build_user_features(users, behaviors, labels)
    X, _ = get_feature_matrix(state.features_df)
    churn_probs, risk_levels = state.churn_model.predict_risk_level(X, days=30)
    state.features_df["churn_prob_30d"] = churn_probs
    state.features_df["risk_level"] = risk_levels
    state.features_df["churn_reason"] = state.churn_model.attribute_churn_reason(X)
    cluster_ids = state.clustering_model.predict(X)
    state.features_df["cluster_id"] = cluster_ids
    state.features_df["cluster_name"] = state.features_df["cluster_id"].map(CLUSTER_NAMES)
    state.initialized = True


@app.get("/")
def read_root():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/system/init")
def init_system(n_users: int = 1000, days: int = 30):
    print(f"初始化系统: {n_users} 用户, {days} 天数据")
    generate_all_data(n_users=n_users, days=days)
    users, behaviors, labels = load_raw_data()
    features = build_user_features(users, behaviors, labels)
    X, feature_names = get_feature_matrix(features)
    churn_model = ChurnPredictor(model_type="gb")
    metrics = churn_model.train(X, labels["is_churned_30d"].values)
    churn_model.save(os.path.join(MODEL_DIR, "churn_model.pkl"))
    clustering = UserClustering(n_clusters=6)
    cluster_labels, silhouette = clustering.fit(X, feature_names)
    clustering.save(os.path.join(MODEL_DIR, "clustering_model.pkl"))
    load_system()
    return {
        "status": "success",
        "n_users": n_users,
        "days": days,
        "churn_model_metrics": metrics,
        "clustering_silhouette": silhouette,
    }


@app.get("/api/system/status")
def system_status():
    if not state.initialized:
        return {"initialized": False}
    return {
        "initialized": True,
        "total_users": len(state.features_df),
        "churn_model_type": state.churn_model.model_type if state.churn_model else None,
        "n_clusters": state.clustering_model.n_clusters if state.clustering_model else 0,
        "ab_experiments": len(state.ab_manager.list_experiments()) if state.ab_manager else 0,
    }


@app.get("/api/users")
def list_users(page: int = 1, page_size: int = 20, risk_level: str = None, cluster: str = None):
    ensure_initialized()
    df = state.features_df.copy()
    if risk_level:
        df = df[df["risk_level"] == risk_level]
    if cluster:
        df = df[df["cluster_name"] == cluster]
    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_data = df.iloc[start:end]
    users = []
    for _, row in page_data.iterrows():
        users.append({
            "user_id": row["user_id"],
            "user_hash": row["user_hash"],
            "age_group": row["age_group"],
            "city_level": row["city_level"],
            "register_days": int(row["register_days"]),
            "total_pay_amount": float(row["total_pay_amount"]),
            "is_vip": bool(row["is_vip"]),
            "churn_prob": round(float(row["churn_prob_30d"]), 4),
            "risk_level": row["risk_level"],
            "churn_reason": row["churn_reason"],
            "cluster_name": row["cluster_name"],
            "active_days_window": int(row["active_days_window"]),
            "total_interaction": int(row["total_interaction"]),
            "prefer_categories": row["prefer_categories"],
        })
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "data": users,
    }


@app.get("/api/users/{user_id}")
def get_user_profile(user_id: str):
    ensure_initialized()
    df = state.features_df[state.features_df["user_id"] == user_id]
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="用户不存在")
    row = df.iloc[0]
    return {
        "user_id": row["user_id"],
        "user_hash": row["user_hash"],
        "profile": {
            "age_group": row["age_group"],
            "city_level": row["city_level"],
            "register_days": int(row["register_days"]),
            "total_pay_amount": float(row["total_pay_amount"]),
            "is_vip": bool(row["is_vip"]),
            "prefer_categories": row["prefer_categories"].split(",") if row["prefer_categories"] else [],
        },
        "behavior": {
            "avg_session_duration": round(float(row["avg_session_duration"]), 1),
            "active_days_window": int(row["active_days_window"]),
            "total_browse": int(row["total_browse"]),
            "total_click": int(row["total_click"]),
            "total_like": int(row["total_like"]),
            "total_interaction": int(row["total_interaction"]),
            "click_through_rate": round(float(row["click_through_rate"]), 4),
            "like_rate": round(float(row["like_rate"]), 4),
            "sessions_7d": int(row["sessions_7d"]),
            "sessions_14d": int(row["sessions_14d"]),
            "decay_ratio": round(float(row["decay_ratio"]), 4),
        },
        "churn_prediction": {
            "churn_prob_30d": round(float(row["churn_prob_30d"]), 4),
            "risk_level": row["risk_level"],
            "churn_reason": row["churn_reason"],
            "days_since_last_active": int(row["days_since_last_active"]),
            "churn_label": row["churn_label"],
        },
        "cluster": {
            "cluster_id": int(row["cluster_id"]),
            "cluster_name": row["cluster_name"],
        },
    }


@app.get("/api/users/{user_id}/recall")
def get_user_recall_plan(user_id: str):
    ensure_initialized()
    df = state.features_df[state.features_df["user_id"] == user_id]
    if len(df) == 0:
        raise HTTPException(status_code=404, detail="用户不存在")
    row = df.iloc[0]
    user_profile = row.to_dict()
    churn_info = {
        "churn_prob": float(row["churn_prob_30d"]),
        "risk_level": row["risk_level"],
        "churn_reason": row["churn_reason"],
    }
    cluster_info = {
        "cluster_id": int(row["cluster_id"]),
        "cluster_name": row["cluster_name"],
    }
    plan = state.recall_engine.generate_recall_plan(user_profile, churn_info, cluster_info)
    return plan


@app.get("/api/statistics/overview")
def get_overview_stats():
    ensure_initialized()
    df = state.features_df
    total = len(df)
    risk_dist = df["risk_level"].value_counts().to_dict()
    cluster_dist = df["cluster_name"].value_counts().to_dict()
    reason_dist = df["churn_reason"].value_counts().to_dict()
    churn_rate = (df["churn_label"] == "深度流失").sum() / total * 100
    avg_pay = df["total_pay_amount"].mean()
    vip_count = df["is_vip"].sum()
    return {
        "total_users": total,
        "churn_rate": round(churn_rate, 2),
        "avg_pay_amount": round(avg_pay, 2),
        "vip_count": int(vip_count),
        "vip_rate": round(vip_count / total * 100, 2),
        "risk_distribution": risk_dist,
        "cluster_distribution": cluster_dist,
        "churn_reason_distribution": reason_dist,
    }


@app.get("/api/statistics/churn")
def get_churn_stats():
    ensure_initialized()
    df = state.features_df
    by_age = df.groupby("age_group")["churn_prob_30d"].mean().round(4).to_dict()
    by_city = df.groupby("city_level")["churn_prob_30d"].mean().round(4).to_dict()
    feature_importance = state.churn_model.feature_importance_ or {}
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        "churn_prob_by_age_group": by_age,
        "churn_prob_by_city_level": by_city,
        "top_feature_importance": {k: round(v, 4) for k, v in top_features},
    }


@app.get("/api/clusters")
def list_clusters():
    ensure_initialized()
    clusters = []
    for cid in range(state.clustering_model.n_clusters):
        profile = state.clustering_model.get_cluster_profile(cid)
        name = state.clustering_model.get_cluster_name(cid)
        clusters.append({
            "cluster_id": cid,
            "cluster_name": name,
            "size": profile.get("size", 0),
            "size_ratio": round(profile.get("size_ratio", 0), 4),
            "top_features": profile.get("top_features", {}),
        })
    return clusters


@app.post("/api/recall/batch")
def batch_recall(top_n: int = 100, risk_filter: str = None):
    ensure_initialized()
    risk_list = None
    if risk_filter:
        risk_list = [r.strip() for r in risk_filter.split(",")]
    plans = state.recall_engine.batch_generate_recalls(
        state.features_df, top_n=top_n, risk_filter=risk_list
    )
    return {
        "count": len(plans),
        "plans": plans,
    }


@app.get("/api/ab/experiments")
def list_ab_experiments():
    ensure_initialized()
    return {"experiments": state.ab_manager.list_experiments()}


@app.post("/api/ab/experiments/{exp_id}/start")
def start_exp(exp_id: str):
    ensure_initialized()
    if state.ab_manager.start_experiment(exp_id):
        return {"status": "started"}
    raise HTTPException(status_code=404, detail="实验不存在")


@app.post("/api/ab/experiments/{exp_id}/stop")
def stop_exp(exp_id: str):
    ensure_initialized()
    if state.ab_manager.stop_experiment(exp_id):
        return {"status": "stopped"}
    raise HTTPException(status_code=404, detail="实验不存在")


@app.get("/api/ab/experiments/{exp_id}/results")
def get_exp_results(exp_id: str):
    ensure_initialized()
    exp = state.ab_manager.get_experiment(exp_id)
    if not exp:
        raise HTTPException(status_code=404, detail="实验不存在")
    return {
        "experiment_id": exp_id,
        "name": exp.name,
        "status": exp.status,
        "results": exp.get_results(),
    }


@app.post("/api/ab/experiments/{exp_id}/assign/{user_id}")
def assign_user_to_exp(exp_id: str, user_id: str):
    ensure_initialized()
    variant = state.ab_manager.assign_user(exp_id, user_id)
    if variant is None:
        raise HTTPException(status_code=400, detail="实验未运行或不存在")
    return {"user_id": user_id, "variant_id": variant}


@app.post("/api/ab/experiments/{exp_id}/metric/{variant_id}/{metric_name}")
def record_exp_metric(exp_id: str, variant_id: str, metric_name: str, value: int = 1):
    ensure_initialized()
    if state.ab_manager.record_metric(exp_id, variant_id, metric_name, value):
        return {"status": "recorded"}
    raise HTTPException(status_code=404, detail="实验不存在")


if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
