import os
import sys
import json
import random
import hashlib
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONTENT_LIB_FILE = os.path.join(DATA_DIR, "content_library.json")
RECOMMEND_CONFIG_FILE = os.path.join(DATA_DIR, "recommend_config.json")

CONTENT_CATEGORIES = ["科技", "娱乐", "体育", "财经", "教育", "生活", "游戏", "美食", "旅游", "时尚"]
CONTENT_AUTHORS = ["科技达人", "娱乐星探", "体育评论员", "财经观察家", "教育专家", "生活家", "游戏博主", "美食家", "旅行家", "时尚博主"]
CONTENT_TOPICS = {
    "科技": ["人工智能", "5G通信", "智能手机", "芯片技术", "新能源汽车", "元宇宙", "区块链", "智能家居"],
    "娱乐": ["电影推荐", "明星八卦", "综艺节目", "音乐排行", "动漫推荐", "游戏资讯", "影视评测", "演唱会"],
    "体育": ["足球", "篮球", "网球", "游泳", "健身", "电子竞技", "马拉松", "极限运动"],
    "财经": ["股票投资", "理财知识", "房产市场", "创业故事", "经济政策", "企业动态", "数字货币", "保险"],
    "教育": ["考研攻略", "职业技能", "英语学习", "编程入门", "育儿知识", "留学申请", "公务员考试", "兴趣培养"],
    "生活": ["健康养生", "家居装修", "穿搭技巧", "情感心理", "职场发展", "亲子关系", "社交技巧", "时间管理"],
    "游戏": ["手游推荐", "端游评测", "游戏攻略", "电竞赛事", "游戏主播", "新游上线", "怀旧游戏", "独立游戏"],
    "美食": ["家常菜", "烘焙甜点", "探店测评", "减脂餐", "地方特色", "咖啡茶饮", "日料韩餐", "西餐做法"],
    "旅游": ["国内游", "出境游", "自驾游", "海岛度假", "古镇文化", "自然风光", "美食之旅", "小众目的地"],
    "时尚": ["潮流穿搭", "美妆护肤", "奢侈品", "街拍", "品牌资讯", "发型设计", "珠宝配饰", "男士时尚"],
}

PUSH_STRATEGIES = {
    "content_only": {
        "name": "纯内容推送",
        "description": "仅推送用户感兴趣的内容，不附带优惠券",
        "coupon_type": "no_coupon",
        "suitable_for": ["高活跃度用户", "内容偏好型用户", "低价格敏感度用户"],
    },
    "content_coupon": {
        "name": "内容+优惠券组合",
        "description": "推送精选内容同时附带优惠券，提升转化",
        "suitable_for": ["中度活跃用户", "价格敏感型用户", "潜在付费用户"],
    },
}

DEFAULT_BLOCKED_WORDS = ["广告", "推广", "诈骗", "虚假", "低俗", "暴力", "色情", "赌博"]
DEFAULT_BLOCKED_CATEGORIES = []


class ContentItem:
    def __init__(self, content_id, title, category, author, topic, tags=None, quality_score=0.7):
        self.content_id = content_id
        self.title = title
        self.category = category
        self.author = author
        self.topic = topic
        self.tags = tags or []
        self.quality_score = quality_score
        self.publish_time = datetime.now().isoformat()

    def to_dict(self):
        return {
            "content_id": self.content_id,
            "title": self.title,
            "category": self.category,
            "author": self.author,
            "topic": self.topic,
            "tags": self.tags,
            "quality_score": self.quality_score,
            "publish_time": self.publish_time,
        }


class ContentLibrary:
    def __init__(self):
        self.contents = {}
        self.category_index = defaultdict(list)
        self.author_index = defaultdict(list)
        self.topic_index = defaultdict(list)
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(CONTENT_LIB_FILE):
            self._load()
        else:
            self._generate_default_library()
            self._save()

    def _load(self):
        with open(CONTENT_LIB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            content = ContentItem(
                content_id=item["content_id"],
                title=item["title"],
                category=item["category"],
                author=item["author"],
                topic=item["topic"],
                tags=item.get("tags", []),
                quality_score=item.get("quality_score", 0.7),
            )
            self.contents[content.content_id] = content
            self.category_index[content.category].append(content.content_id)
            self.author_index[content.author].append(content.content_id)
            self.topic_index[content.topic].append(content.content_id)

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        data = [c.to_dict() for c in self.contents.values()]
        with open(CONTENT_LIB_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _generate_default_library(self, n_per_category=20):
        for category in CONTENT_CATEGORIES:
            topics = CONTENT_TOPICS.get(category, ["精选"])
            for i in range(n_per_category):
                topic = random.choice(topics)
                author = random.choice(CONTENT_AUTHORS)
                title = self._generate_title(category, topic, i)
                content_id = hashlib.md5(f"{category}_{topic}_{i}".encode()).hexdigest()[:12]
                tags = [category, topic, author] + random.sample(topics, min(2, len(topics)))
                quality_score = round(random.uniform(0.5, 0.98), 3)
                content = ContentItem(content_id, title, category, author, topic, tags, quality_score)
                self.contents[content_id] = content
                self.category_index[category].append(content_id)
                self.author_index[author].append(content_id)
                self.topic_index[topic].append(content_id)

    def _generate_title(self, category, topic, index):
        templates = [
            "深度解析：{topic}的未来发展趋势",
            "{topic}入门指南：新手必看的{count}个技巧",
            "专家解读：{category}领域最新动态",
            "为什么{topic}突然火了？原因在这里",
            "{author}亲授：{topic}的正确打开方式",
            "盘点{topic}的{count}大误区，你中招了吗？",
            "干货分享：{topic}实战经验总结",
            "收藏！{category}爱好者不可错过的{topic}合集",
        ]
        template = random.choice(templates)
        return template.format(topic=topic, category=category, author=random.choice(CONTENT_AUTHORS), count=random.randint(3, 10))

    def get_by_category(self, category):
        return [self.contents[cid] for cid in self.category_index.get(category, [])]

    def get_by_author(self, author):
        return [self.contents[cid] for cid in self.author_index.get(author, [])]

    def get_by_topic(self, topic):
        return [self.contents[cid] for cid in self.topic_index.get(topic, [])]

    def search(self, keyword):
        keyword = keyword.lower()
        results = []
        for content in self.contents.values():
            if (keyword in content.title.lower() or
                keyword in content.category.lower() or
                keyword in content.topic.lower() or
                any(keyword in tag.lower() for tag in content.tags)):
                results.append(content)
        return results

    def get_all_categories(self):
        return CONTENT_CATEGORIES

    def get_topics_by_category(self, category):
        return CONTENT_TOPICS.get(category, [])

    def add_content(self, content_item):
        self.contents[content_item.content_id] = content
        self.category_index[content.category].append(content.content_id)
        self.author_index[content.author].append(content.content_id)
        self.topic_index[content.topic].append(content.content_id)
        self._save()


class RecommendConfig:
    def __init__(self):
        self.blocked_words = list(DEFAULT_BLOCKED_WORDS)
        self.blocked_categories = list(DEFAULT_BLOCKED_CATEGORIES)
        self.min_quality_score = 0.5
        self.max_recommend_count = 5
        self.enable_personalized = True
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(RECOMMEND_CONFIG_FILE):
            self._load()
        else:
            self._save()

    def _load(self):
        with open(RECOMMEND_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.blocked_words = data.get("blocked_words", list(DEFAULT_BLOCKED_WORDS))
        self.blocked_categories = data.get("blocked_categories", list(DEFAULT_BLOCKED_CATEGORIES))
        self.min_quality_score = data.get("min_quality_score", 0.5)
        self.max_recommend_count = data.get("max_recommend_count", 5)
        self.enable_personalized = data.get("enable_personalized", True)

    def _save(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        data = {
            "blocked_words": self.blocked_words,
            "blocked_categories": self.blocked_categories,
            "min_quality_score": self.min_quality_score,
            "max_recommend_count": self.max_recommend_count,
            "enable_personalized": self.enable_personalized,
        }
        with open(RECOMMEND_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_blocked_word(self, word):
        if word not in self.blocked_words:
            self.blocked_words.append(word)
            self._save()
            return True
        return False

    def remove_blocked_word(self, word):
        if word in self.blocked_words:
            self.blocked_words.remove(word)
            self._save()
            return True
        return False

    def add_blocked_category(self, category):
        if category not in self.blocked_categories and category in CONTENT_CATEGORIES:
            self.blocked_categories.append(category)
            self._save()
            return True
        return False

    def remove_blocked_category(self, category):
        if category in self.blocked_categories:
            self.blocked_categories.remove(category)
            self._save()
            return True
        return False

    def update_config(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._save()

    def is_content_blocked(self, content):
        if content.category in self.blocked_categories:
            return True
        for word in self.blocked_words:
            if word.lower() in content.title.lower() or any(word.lower() in tag.lower() for tag in content.tags):
                return True
        if content.quality_score < self.min_quality_score:
            return True
        return False

    def to_dict(self):
        return {
            "blocked_words": self.blocked_words,
            "blocked_categories": self.blocked_categories,
            "min_quality_score": self.min_quality_score,
            "max_recommend_count": self.max_recommend_count,
            "enable_personalized": self.enable_personalized,
        }


class PersonalizedContentRecommender:
    def __init__(self):
        self.content_library = ContentLibrary()
        self.config = RecommendConfig()

    def _parse_user_preferences(self, user_profile):
        prefer_categories = user_profile.get("prefer_categories", "")
        if isinstance(prefer_categories, str) and prefer_categories:
            category_list = [c.strip() for c in prefer_categories.split(",") if c.strip()]
        else:
            category_list = random.sample(CONTENT_CATEGORIES, 3)

        category_weights = {}
        for i, cat in enumerate(category_list):
            category_weights[cat] = max(0.3, 1.0 - i * 0.15)

        return {
            "categories": category_list,
            "category_weights": category_weights,
            "total_pay": user_profile.get("total_pay_amount", 0),
            "is_vip": user_profile.get("is_vip", False),
            "active_days": user_profile.get("active_days_window", 7),
            "click_through_rate": user_profile.get("click_through_rate", 0.2),
            "total_interaction": user_profile.get("total_interaction", 0),
        }

    def _determine_push_strategy(self, user_prefs, churn_reason):
        total_pay = user_prefs["total_pay"]
        is_vip = user_prefs["is_vip"]
        ctr = user_prefs["click_through_rate"]
        active_days = user_prefs["active_days"]

        content_pref_score = ctr * 0.6 + (active_days / 30) * 0.4

        pay_sensitivity = 0
        if total_pay > 500:
            pay_sensitivity = 0.2
        elif total_pay > 200:
            pay_sensitivity = 0.5
        elif total_pay > 50:
            pay_sensitivity = 0.7
        else:
            pay_sensitivity = 0.9

        if churn_reason in ["内容不合预期", "互动减少"]:
            content_pref_score = min(1.0, content_pref_score + 0.3)
        elif churn_reason in ["付费意愿低"]:
            pay_sensitivity = min(1.0, pay_sensitivity + 0.2)

        if content_pref_score > 0.6 and pay_sensitivity < 0.4:
            strategy = "content_only"
        elif pay_sensitivity > 0.6 or is_vip:
            strategy = "content_coupon"
        else:
            strategy = "content_coupon" if random.random() < 0.6 else "content_only"

        return strategy

    def _select_coupon_type(self, user_prefs, churn_reason):
        total_pay = user_prefs["total_pay"]
        is_vip = user_prefs["is_vip"]

        if is_vip or total_pay > 500:
            return "vip_discount"
        elif total_pay > 200:
            return "cash_coupon"
        elif total_pay > 50:
            return "points_bonus"
        else:
            return "exclusive_content"

    def _score_content(self, content, user_prefs):
        if self.config.is_content_blocked(content):
            return 0

        score = 0.0

        category_weight = user_prefs["category_weights"].get(content.category, 0.1)
        score += category_weight * 0.4

        quality_weight = content.quality_score
        score += quality_weight * 0.3

        topic_match = 0
        prefer_topics = []
        for cat in user_prefs["categories"]:
            prefer_topics.extend(CONTENT_TOPICS.get(cat, []))
        if content.topic in prefer_topics:
            topic_match = 0.3
        score += topic_match * 0.2

        freshness = random.uniform(0.8, 1.0)
        score += freshness * 0.1

        return score

    def recommend(self, user_profile, churn_info, count=None):
        if count is None:
            count = self.config.max_recommend_count

        user_prefs = self._parse_user_preferences(user_profile)
        churn_reason = churn_info.get("churn_reason", "行为衰减")

        push_strategy = self._determine_push_strategy(user_prefs, churn_reason)

        all_contents = list(self.content_library.contents.values())

        valid_contents = [c for c in all_contents if not self.config.is_content_blocked(c)]

        scored_contents = []
        for content in valid_contents:
            score = self._score_content(content, user_prefs)
            if score > 0:
                scored_contents.append((content, score))

        scored_contents.sort(key=lambda x: x[1], reverse=True)

        selected_contents = []
        used_categories = set()
        for content, score in scored_contents:
            if len(selected_contents) >= count:
                break
            if content.category not in used_categories or len(selected_contents) < count // 2:
                selected_contents.append({
                    "content": content.to_dict(),
                    "match_score": round(score, 4),
                    "match_reason": self._get_match_reason(content, user_prefs),
                })
                used_categories.add(content.category)

        while len(selected_contents) < count and len(scored_contents) > len(selected_contents):
            for content, score in scored_contents[len(selected_contents):]:
                if len(selected_contents) >= count:
                    break
                already = any(c["content"]["content_id"] == content.content_id for c in selected_contents)
                if not already:
                    selected_contents.append({
                        "content": content.to_dict(),
                        "match_score": round(score, 4),
                        "match_reason": self._get_match_reason(content, user_prefs),
                    })

        coupon_type = None
        coupon_info = None
        if push_strategy == "content_coupon":
            coupon_type = self._select_coupon_type(user_prefs, churn_reason)
            coupon_info = self._get_coupon_info(coupon_type)

        primary_content = selected_contents[0] if selected_contents else None
        push_text = self._generate_push_text(primary_content, push_strategy, coupon_info, churn_reason)

        return {
            "user_id": user_profile.get("user_id", ""),
            "push_strategy": push_strategy,
            "strategy_name": PUSH_STRATEGIES[push_strategy]["name"],
            "churn_reason": churn_reason,
            "push_content": push_text,
            "coupon": coupon_info,
            "recommendations": selected_contents,
            "user_preferences": {
                "preferred_categories": user_prefs["categories"][:3],
                "push_strategy_reason": self._get_strategy_reason(push_strategy, user_prefs, churn_reason),
            },
            "generate_time": datetime.now().isoformat(),
        }

    def _get_match_reason(self, content, user_prefs):
        reasons = []
        if content.category in user_prefs["categories"]:
            reasons.append(f"你感兴趣的{content.category}类目")
        if content.quality_score > 0.85:
            reasons.append("高质量内容")
        prefer_topics = []
        for cat in user_prefs["categories"][:2]:
            prefer_topics.extend(CONTENT_TOPICS.get(cat, []))
        if content.topic in prefer_topics:
            reasons.append(f"{content.topic}相关")
        return reasons if reasons else ["精选推荐"]

    def _get_strategy_reason(self, strategy, user_prefs, churn_reason):
        if strategy == "content_only":
            reasons = []
            if user_prefs["click_through_rate"] > 0.3:
                reasons.append("用户内容偏好度高")
            if user_prefs["total_pay"] > 500:
                reasons.append("用户价格敏感度低")
            if churn_reason in ["内容不合预期"]:
                reasons.append("以优质内容重新吸引")
            return reasons if reasons else ["纯内容推送策略"]
        else:
            reasons = []
            if user_prefs["total_pay"] < 200:
                reasons.append("用户价格敏感度较高")
            if churn_reason in ["付费意愿低"]:
                reasons.append("优惠券刺激转化")
            if user_prefs["is_vip"]:
                reasons.append("VIP专属福利")
            return reasons if reasons else ["内容+优惠券组合策略"]

    def _get_coupon_info(self, coupon_type):
        COUPONS = {
            "vip_discount": {"name": "会员折扣券", "value": "7折", "cost": 15, "validity": "7天"},
            "cash_coupon": {"name": "现金券", "value": "10元", "cost": 10, "validity": "3天"},
            "points_bonus": {"name": "积分奖励", "value": "200积分", "cost": 5, "validity": "15天"},
            "exclusive_content": {"name": "专属内容", "value": "7天免费", "cost": 8, "validity": "7天"},
        }
        return COUPONS.get(coupon_type, COUPONS["exclusive_content"])

    def _generate_push_text(self, primary_content, strategy, coupon_info, churn_reason):
        if primary_content:
            content = primary_content["content"]
            category = content["category"]
            topic = content["topic"]
        else:
            category = "精选"
            topic = "内容"

        templates = {
            "content_only": [
                f"🔥 你感兴趣的{category}又上新了，{topic}别错过！",
                f"📌 精选{category}内容：{topic}深度解析",
                f"✨ {topic}新动态，点击查看详情",
            ],
            "content_coupon": [
                f"🎁 {coupon_info['value']}已到账，顺便看看{topic}精选内容",
                f"💰 送你{coupon_info['value']}，还有{category}好内容等你发现",
                f"🎉 惊喜好礼 + {topic}推荐，点击一键领取",
            ],
            "comeback": [
                f"👋 好久不见，为你准备了{category}精选内容",
                f"🌟 离开的日子，{topic}有新动态哦",
                f"💫 欢迎回来，{category}专区更新了",
            ],
        }

        if churn_reason in ["内容不合预期", "互动减少"]:
            template_group = "content_only" if strategy == "content_only" else "content_coupon"
        else:
            template_group = "content_coupon" if strategy == "content_coupon" else "comeback"

        primary = random.choice(templates[template_group])
        alternatives = [t for t in templates[template_group] if t != primary][:2]

        return {
            "primary": primary,
            "alternatives": alternatives,
            "content_preview": content["title"] if primary_content else "",
        }

    def batch_recommend(self, user_features_df, churn_df=None, count=None):
        results = []
        for _, row in user_features_df.iterrows():
            user_profile = row.to_dict()
            churn_info = {
                "churn_prob": row.get("churn_prob_30d", 0),
                "risk_level": row.get("risk_level", "中风险"),
                "churn_reason": row.get("churn_reason", "行为衰减"),
            }
            rec = self.recommend(user_profile, churn_info, count)
            results.append(rec)
        return results

    def get_config(self):
        return self.config.to_dict()

    def update_config(self, **kwargs):
        self.config.update_config(**kwargs)
        return self.config.to_dict()

    def add_blocked_word(self, word):
        return self.config.add_blocked_word(word)

    def remove_blocked_word(self, word):
        return self.config.remove_blocked_word(word)

    def add_blocked_category(self, category):
        return self.config.add_blocked_category(category)

    def remove_blocked_category(self, category):
        return self.config.remove_blocked_category(category)

    def get_content_library(self, category=None, page=1, page_size=20):
        contents = list(self.content_library.contents.values())
        if category:
            contents = [c for c in contents if c.category == category]
        total = len(contents)
        start = (page - 1) * page_size
        end = start + page_size
        page_data = [c.to_dict() for c in contents[start:end]]
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "data": page_data,
        }

    def get_categories(self):
        return CONTENT_CATEGORIES

    def get_topics(self, category):
        return CONTENT_TOPICS.get(category, [])


def get_recommender():
    return PersonalizedContentRecommender()


if __name__ == "__main__":
    recommender = PersonalizedContentRecommender()
    print("个性化内容推荐引擎初始化完成")
    print(f"内容库大小: {len(recommender.content_library.contents)} 条")
    print(f"支持分类: {len(CONTENT_CATEGORIES)} 个")

    sample_user = {
        "user_id": "test_user_001",
        "prefer_categories": "科技,财经,教育",
        "total_pay_amount": 350,
        "is_vip": True,
        "active_days_window": 5,
        "click_through_rate": 0.25,
        "total_interaction": 150,
    }
    churn_info = {
        "churn_prob": 0.65,
        "risk_level": "高风险",
        "churn_reason": "内容不合预期",
    }

    result = recommender.recommend(sample_user, churn_info)
    print(f"\n推荐结果:")
    print(f"  用户ID: {result['user_id']}")
    print(f"  推送策略: {result['strategy_name']}")
    print(f"  主推文案: {result['push_content']['primary']}")
    if result["coupon"]:
        print(f"  优惠券: {result['coupon']['name']} ({result['coupon']['value']})")
    print(f"  推荐内容数: {len(result['recommendations'])}")
    for i, rec in enumerate(result['recommendations'][:3]):
        print(f"    {i+1}. [{rec['content']['category']}] {rec['content']['title'][:50]}... (匹配度: {rec['match_score']})")

    print(f"\n配置信息:")
    config = recommender.get_config()
    print(f"  屏蔽词数量: {len(config['blocked_words'])}")
    print(f"  屏蔽分类: {config['blocked_categories']}")
    print(f"  最低质量分: {config['min_quality_score']}")
