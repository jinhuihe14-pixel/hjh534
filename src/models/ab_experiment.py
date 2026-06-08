import os
import sys
import json
import uuid
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
EXPERIMENT_FILE = os.path.join(DATA_DIR, "ab_experiments.json")


class ABExperiment:
    def __init__(self, name, description="", variants=None):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.description = description
        self.variants = variants or []
        self.status = "created"
        self.created_at = datetime.now().isoformat()
        self.started_at = None
        self.ended_at = None
        self.metrics = defaultdict(lambda: defaultdict(int))
        self.user_assignments = {}

    def add_variant(self, variant_id, variant_name, traffic_weight=1.0, config=None):
        self.variants.append({
            "id": variant_id,
            "name": variant_name,
            "traffic_weight": traffic_weight,
            "config": config or {},
        })

    def assign_user(self, user_id):
        if user_id in self.user_assignments:
            return self.user_assignments[user_id]
        total_weight = sum(v["traffic_weight"] for v in self.variants)
        import random
        r = random.uniform(0, total_weight)
        cumulative = 0
        assigned = None
        for v in self.variants:
            cumulative += v["traffic_weight"]
            if r <= cumulative:
                assigned = v
                break
        if assigned is None:
            assigned = self.variants[-1]
        self.user_assignments[user_id] = assigned["id"]
        self.metrics[assigned["id"]]["exposure"] += 1
        return assigned["id"]

    def record_metric(self, variant_id, metric_name, value=1):
        self.metrics[variant_id][metric_name] += value

    def get_results(self):
        results = []
        for v in self.variants:
            vid = v["id"]
            m = self.metrics.get(vid, {})
            exposure = m.get("exposure", 0)
            conversions = m.get("conversion", 0)
            ctr = conversions / exposure if exposure > 0 else 0
            results.append({
                "variant_id": vid,
                "variant_name": v["name"],
                "exposure": exposure,
                "conversions": conversions,
                "conversion_rate": round(ctr * 100, 2),
                "metrics": dict(m),
            })
        return results

    def start(self):
        self.status = "running"
        self.started_at = datetime.now().isoformat()

    def stop(self):
        self.status = "stopped"
        self.ended_at = datetime.now().isoformat()

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "variants": self.variants,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "metrics": {k: dict(v) for k, v in self.metrics.items()},
            "user_assignments": self.user_assignments,
        }

    @classmethod
    def from_dict(cls, data):
        exp = cls(data["name"], data.get("description", ""))
        exp.id = data["id"]
        exp.variants = data["variants"]
        exp.status = data["status"]
        exp.created_at = data["created_at"]
        exp.started_at = data.get("started_at")
        exp.ended_at = data.get("ended_at")
        exp.metrics = defaultdict(lambda: defaultdict(int))
        for k, v in data.get("metrics", {}).items():
            exp.metrics[k] = defaultdict(int, v)
        exp.user_assignments = data.get("user_assignments", {})
        return exp


class ABExperimentManager:
    def __init__(self):
        self.experiments = {}
        self._load()

    def _load(self):
        if os.path.exists(EXPERIMENT_FILE):
            with open(EXPERIMENT_FILE, "r") as f:
                data = json.load(f)
            for exp_data in data:
                exp = ABExperiment.from_dict(exp_data)
                self.experiments[exp.id] = exp

    def _save(self):
        data = [exp.to_dict() for exp in self.experiments.values()]
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(EXPERIMENT_FILE, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def create_experiment(self, name, description="", variants=None):
        exp = ABExperiment(name, description, variants)
        self.experiments[exp.id] = exp
        self._save()
        return exp

    def get_experiment(self, exp_id):
        return self.experiments.get(exp_id)

    def list_experiments(self):
        return [exp.to_dict() for exp in self.experiments.values()]

    def assign_user(self, exp_id, user_id):
        exp = self.get_experiment(exp_id)
        if not exp or exp.status != "running":
            return None
        variant_id = exp.assign_user(user_id)
        self._save()
        return variant_id

    def record_metric(self, exp_id, variant_id, metric_name, value=1):
        exp = self.get_experiment(exp_id)
        if not exp:
            return False
        exp.record_metric(variant_id, metric_name, value)
        self._save()
        return True

    def start_experiment(self, exp_id):
        exp = self.get_experiment(exp_id)
        if exp:
            exp.start()
            self._save()
            return True
        return False

    def stop_experiment(self, exp_id):
        exp = self.get_experiment(exp_id)
        if exp:
            exp.stop()
            self._save()
            return True
        return False

    def get_winner(self, exp_id):
        exp = self.get_experiment(exp_id)
        if not exp:
            return None
        results = exp.get_results()
        if not results:
            return None
        best = max(results, key=lambda x: x["conversion_rate"])
        return best


def create_default_experiments():
    manager = ABExperimentManager()
    if len(manager.list_experiments()) > 0:
        return manager
    exp1 = manager.create_experiment(
        name="召回文案效果测试",
        description="测试不同召回推送文案的转化效果",
    )
    exp1.add_variant("control", "对照组-通用文案", traffic_weight=1.0)
    exp1.add_variant("personalized", "实验组-个性化文案", traffic_weight=1.0)
    exp1.add_variant("urgency", "实验组-紧迫感文案", traffic_weight=1.0)
    exp2 = manager.create_experiment(
        name="优惠券类型测试",
        description="测试不同优惠券类型的召回转化",
    )
    exp2.add_variant("no_coupon", "对照组-无优惠券", traffic_weight=1.0)
    exp2.add_variant("cash", "实验组-现金券", traffic_weight=1.0)
    exp2.add_variant("vip", "实验组-VIP折扣", traffic_weight=1.0)
    manager._save()
    print(f"已创建 {len(manager.list_experiments())} 个默认A/B实验")
    return manager


if __name__ == "__main__":
    manager = create_default_experiments()
    for exp in manager.list_experiments():
        print(f"  [{exp['status']}] {exp['name']} - {len(exp['variants'])}个版本")
