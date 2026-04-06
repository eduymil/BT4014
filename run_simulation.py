import argparse
import numpy as np
import pandas as pd
import time
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading data...")

data_path_news = Path("dataset") / "news.tsv"
data_path_user_interaction = Path("dataset") / "behaviors.tsv"
data_path_news_embeddings = Path("dataset") / "article_embeddings.npy"

news_articles = pd.read_csv(data_path_news, sep="\t", header=None)
news_articles.columns = ['News_ID', 'Category', 'Subcategory', 'Title', 'Abstract', 'URL', 'Title Entities', 'Abstract Entities']

users_interaction = pd.read_csv(data_path_user_interaction, sep="\t", header=None)
users_interaction.columns = ['No.', 'User_ID', 'Time_Stamp', 'History', 'Impression']

article_embeddings = np.load(data_path_news_embeddings)
EMB_DIM = article_embeddings.shape[1]

article_embedding_dict = {nid: emb for nid, emb in zip(news_articles["News_ID"], article_embeddings)}

# Dictionary mapping Article ID to its Category (Crucial for Non-Contextual)
news_to_category = dict(zip(news_articles['News_ID'], news_articles['Category']))
all_categories = news_articles['Category'].unique().tolist()

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("Feature engineering...")

def get_history_count(history):
    if pd.isna(history) or history == "": return 0
    return len(history.split())

users_interaction['history_count'] = users_interaction['History'].apply(get_history_count)
users_interaction['history_count_norm'] = users_interaction['history_count'] / users_interaction['history_count'].max()

def get_category_freq(history):
    freq = defaultdict(int)
    if pd.isna(history) or history == "": return np.zeros(len(all_categories))
    articles = history.split()
    for article in articles:
        cat = news_to_category.get(article)
        if cat: freq[cat] += 1
    arr = np.array([freq[c] for c in all_categories], dtype=float)
    if len(articles) > 0: arr /= len(articles)
    return arr

users_interaction['category_freq'] = users_interaction['History'].apply(get_category_freq)

users_interaction['Time_Stamp'] = pd.to_datetime(users_interaction['Time_Stamp'])
users_interaction['hour_norm'] = users_interaction['Time_Stamp'].dt.hour / 23.0
users_interaction['day_norm'] = users_interaction['Time_Stamp'].dt.dayofweek / 6.0

def get_user_vector(history):
    vecs = [article_embedding_dict[a] for a in history.split() if a in article_embedding_dict]
    if len(vecs) == 0: return np.zeros(EMB_DIM)
    return np.mean(vecs, axis=0)

users_interaction['user_vector'] = users_interaction['History'].fillna('').apply(get_user_vector)

# ==========================================
# 3. PCA & VECTORIZED FEATURE BUILDING
# ==========================================
print("Running PCA & Vectorizing User Features...")

article_ids = list(article_embedding_dict.keys())
X_articles = np.array([article_embedding_dict[aid] for aid in article_ids])
pca = PCA(n_components=64, random_state=42)
X_articles_64 = pca.fit_transform(X_articles)
article_embedding_dict_64 = {aid: X_articles_64[i] for i, aid in enumerate(article_ids)}

X_users_pca = pca.transform(np.vstack(users_interaction["user_vector"].values))
time_feats = users_interaction[["hour_norm", "day_norm"]].values
hist_feat = users_interaction[["history_count_norm"]].values
cat_feats = np.vstack(users_interaction["category_freq"].values)

user_features_matrix = np.hstack([X_users_pca, time_feats, hist_feat, cat_feats]).astype(np.float32)
users_interaction["user_features"] = list(user_features_matrix)

# ==========================================
# 4. IMPRESSION PARSING
# ==========================================
def parse_impressions(imp_str):
    if pd.isna(imp_str) or imp_str == "": return [], None
    candidates, clicked = [], None
    for item in imp_str.split():
        if "-" not in item: continue
        aid, label = item.rsplit("-", 1)
        candidates.append(aid)
        if label == "1": clicked = aid
    return candidates, clicked

users_interaction[["candidate_ids", "clicked_id"]] = users_interaction["Impression"].apply(
    lambda x: pd.Series(parse_impressions(x))
)

def make_context(user_features, article_emb):
    return np.concatenate([user_features, article_emb])

# ==========================================
# 5A. NON-CONTEXTUAL ALGORITHMS (Category Tracking)
# ==========================================
class BaseCategoryBandit:
    def __init__(self, name):
        self.name = name
        self.counts = defaultdict(int)
        self.values = defaultdict(float)
        self.last_cat = None

    def update(self, reward):
        if self.last_cat is None: return
        self.counts[self.last_cat] += 1
        n = self.counts[self.last_cat]
        val = self.values[self.last_cat]
        self.values[self.last_cat] = ((n - 1) / n) * val + (1 / n) * reward

class EpsilonGreedy:
    def __init__(self, epsilon=0.1):
        self.base = BaseCategoryBandit("Epsilon Greedy (Category)")
        self.epsilon = epsilon

    def select_arm(self, candidates, news_to_cat, **kwargs):
        if np.random.rand() < self.epsilon:
            chosen = np.random.choice(candidates)
        else:
            best_score = -np.inf
            chosen = candidates[0]
            for aid in candidates:
                cat = news_to_cat.get(aid, "Unknown")
                if self.base.values[cat] > best_score:
                    best_score = self.base.values[cat]
                    chosen = aid
        self.base.last_cat = news_to_cat.get(chosen, "Unknown")
        return chosen

    def update(self, reward): self.base.update(reward)

class DecayingEpsilon:
    def __init__(self):
        self.base = BaseCategoryBandit("Decaying Epsilon (Category)")
        self.t = 1

    def select_arm(self, candidates, news_to_cat, **kwargs):
        epsilon = 1.0 / np.sqrt(self.t)
        if np.random.rand() < epsilon:
            chosen = np.random.choice(candidates)
        else:
            best_score = -np.inf
            chosen = candidates[0]
            for aid in candidates:
                cat = news_to_cat.get(aid, "Unknown")
                if self.base.values[cat] > best_score:
                    best_score = self.base.values[cat]
                    chosen = aid
        self.base.last_cat = news_to_cat.get(chosen, "Unknown")
        self.t += 1
        return chosen

    def update(self, reward): self.base.update(reward)

class Softmax:
    def __init__(self, tau=0.1):
        self.base = BaseCategoryBandit("Softmax (Category)")
        self.tau = tau

    def select_arm(self, candidates, news_to_cat, **kwargs):
        vals = np.array([self.base.values[news_to_cat.get(aid, "Unknown")] for aid in candidates])
        z = vals / self.tau
        exp_z = np.exp(z - np.max(z))
        probs = exp_z / np.sum(exp_z)
        chosen = np.random.choice(candidates, p=probs)
        self.base.last_cat = news_to_cat.get(chosen, "Unknown")
        return chosen

    def update(self, reward): self.base.update(reward)

class UCB1:
    def __init__(self):
        self.base = BaseCategoryBandit("UCB1 (Category)")
        self.total_pulls = 0

    def select_arm(self, candidates, news_to_cat, **kwargs):
        best_score = -np.inf
        chosen = None
        for aid in candidates:
            cat = news_to_cat.get(aid, "Unknown")
            if self.base.counts[cat] == 0:
                score = np.inf
            else:
                score = self.base.values[cat] + np.sqrt(2 * np.log(self.total_pulls) / self.base.counts[cat])
            
            if score > best_score:
                best_score = score
                chosen = aid
                
        self.base.last_cat = news_to_cat.get(chosen, "Unknown")
        return chosen

    def update(self, reward):
        self.total_pulls += 1
        self.base.update(reward)

class ThompsonSampling:
    def __init__(self):
        self.name = "Thompson Sampling (Category)"
        self.successes = defaultdict(lambda: 1) 
        self.failures = defaultdict(lambda: 1)  
        self.last_cat = None

    def select_arm(self, candidates, news_to_cat, **kwargs):
        best_score = -np.inf
        chosen = None
        for aid in candidates:
            cat = news_to_cat.get(aid, "Unknown")
            score = np.random.beta(self.successes[cat], self.failures[cat])
            if score > best_score:
                best_score = score
                chosen = aid
                
        self.last_cat = news_to_cat.get(chosen, "Unknown")
        return chosen

    def update(self, reward):
        if self.last_cat is None: return
        if reward == 1:
            self.successes[self.last_cat] += 1
        else:
            self.failures[self.last_cat] += 1

# ==========================================
# 5B. SHARED CONTEXTUAL ALGORITHMS
# ==========================================
class SharedEpsilonGreedy:
    def __init__(self, d, epsilon=0.1, lr=0.01):
        self.name = "Shared Epsilon Greedy"
        self.theta = np.zeros(d)
        self.epsilon = epsilon
        self.lr = lr
        self.last_context = None

    def select_arm(self, user_features, candidates, emb_dict, **kwargs):
        valid = [aid for aid in candidates if aid in emb_dict]
        if not valid: return None
        if np.random.rand() < self.epsilon:
            chosen = np.random.choice(valid)
            self.last_context = make_context(user_features, emb_dict[chosen])
            return chosen

        scores, contexts = [], []
        for aid in valid:
            x = make_context(user_features, emb_dict[aid])
            scores.append(x @ self.theta)
            contexts.append(x)
        idx = np.argmax(scores)
        self.last_context = contexts[idx]
        return valid[idx]

    def update(self, reward):
        if self.last_context is not None:
            error = reward - (self.last_context @ self.theta)
            self.theta += self.lr * error * self.last_context

class SharedLinUCB:
    def __init__(self, d, alpha=1.0):
        self.name = "Shared LinUCB"
        self.alpha = alpha
        self.A_inv = np.eye(d)
        self.b = np.zeros(d)
        self.theta = np.zeros(d)
        self.last_context = None

    def select_arm(self, user_features, candidates, emb_dict, **kwargs):
        best_score, best_arm, best_context = -np.inf, None, None
        for aid in candidates:
            if aid not in emb_dict: continue
            x = make_context(user_features, emb_dict[aid])
            score = x @ self.theta + self.alpha * np.sqrt(x @ self.A_inv @ x)
            if score > best_score:
                best_score, best_arm, best_context = score, aid, x
        self.last_context = best_context
        return best_arm

    def update(self, reward):
        if self.last_context is None: return
        x = self.last_context
        Ax = self.A_inv @ x
        self.A_inv -= np.outer(Ax, Ax) / (1 + x @ Ax)
        self.b += reward * x
        self.theta = self.A_inv @ self.b

class SharedTS:
    def __init__(self, d, v=0.3):
        self.name = "Shared Thompson Sampling"
        self.A_inv = np.eye(d)
        self.b = np.zeros(d)
        self.mu = np.zeros(d)
        self.v = v
        self.last_context = None

    def select_arm(self, user_features, candidates, emb_dict, **kwargs):
        best_score, best_arm, best_context = -np.inf, None, None
        for aid in candidates:
            if aid not in emb_dict: continue
            x = make_context(user_features, emb_dict[aid])
            var = max((self.v ** 2) * (x @ self.A_inv @ x), 1e-6)
            score = np.random.normal(x @ self.mu, np.sqrt(var))
            if score > best_score:
                best_score, best_arm, best_context = score, aid, x
        self.last_context = best_context
        return best_arm

    def update(self, reward):
        if self.last_context is None: return
        x = self.last_context
        Ax = self.A_inv @ x
        self.A_inv -= np.outer(Ax, Ax) / (1 + x @ Ax)
        self.b += reward * x
        self.mu = self.A_inv @ self.b

# ==========================================
# 5C. DISJOINT CONTEXTUAL ALGORITHMS
# ==========================================
class DisjointEpsilonGreedy:
    def __init__(self, d, epsilon=0.1, lr=0.01):
        self.name = "Disjoint Epsilon Greedy"
        self.d = d
        self.epsilon = epsilon
        self.lr = lr
        self.theta = {}
        self.last_context = None
        self.last_arm = None

    def _init_arm(self, aid):
        if aid not in self.theta:
            self.theta[aid] = np.zeros(self.d)

    def select_arm(self, user_features, candidates, emb_dict, **kwargs):
        valid = [aid for aid in candidates if aid in emb_dict]
        if not valid: return None
        if np.random.rand() < self.epsilon:
            chosen = np.random.choice(valid)
            self._init_arm(chosen)
            self.last_context = make_context(user_features, emb_dict[chosen])
            self.last_arm = chosen
            return chosen

        scores, contexts = [], []
        for aid in valid:
            self._init_arm(aid)
            x = make_context(user_features, emb_dict[aid])
            scores.append(x @ self.theta[aid])
            contexts.append(x)
        idx = np.argmax(scores)
        self.last_context = contexts[idx]
        self.last_arm = valid[idx]
        return valid[idx]

    def update(self, reward):
        if self.last_context is None or self.last_arm is None: return
        aid = self.last_arm
        error = reward - (self.last_context @ self.theta[aid])
        self.theta[aid] += self.lr * error * self.last_context

class DisjointLinUCB:
    def __init__(self, d, alpha=1.0):
        self.name = "Disjoint LinUCB"
        self.alpha = alpha
        self.d = d
        self.A_inv = {}
        self.b = {}
        self.theta = {}
        self.last_context = None
        self.last_arm = None

    def _init_arm(self, aid):
        if aid not in self.A_inv:
            self.A_inv[aid] = np.eye(self.d)
            self.b[aid] = np.zeros(self.d)
            self.theta[aid] = np.zeros(self.d)

    def select_arm(self, user_features, candidates, emb_dict, **kwargs):
        best_score, best_arm, best_context = -np.inf, None, None
        for aid in candidates:
            if aid not in emb_dict: continue
            self._init_arm(aid)
            x = make_context(user_features, emb_dict[aid])
            score = x @ self.theta[aid] + self.alpha * np.sqrt(x @ self.A_inv[aid] @ x)
            if score > best_score:
                best_score, best_arm, best_context = score, aid, x
        self.last_context = best_context
        self.last_arm = best_arm
        return best_arm

    def update(self, reward):
        if self.last_context is None or self.last_arm is None: return
        x = self.last_context
        aid = self.last_arm
        Ax = self.A_inv[aid] @ x
        self.A_inv[aid] -= np.outer(Ax, Ax) / (1 + x @ Ax)
        self.b[aid] += reward * x
        self.theta[aid] = self.A_inv[aid] @ self.b[aid]

class DisjointTS:
    def __init__(self, d, v=0.3):
        self.name = "Disjoint Thompson Sampling"
        self.v = v
        self.d = d
        self.A_inv = {}
        self.b = {}
        self.mu = {}
        self.last_context = None
        self.last_arm = None
        
    def _init_arm(self, aid):
        if aid not in self.A_inv:
            self.A_inv[aid] = np.eye(self.d)
            self.b[aid] = np.zeros(self.d)
            self.mu[aid] = np.zeros(self.d)

    def select_arm(self, user_features, candidates, emb_dict, **kwargs):
        best_score, best_arm, best_context = -np.inf, None, None
        for aid in candidates:
            if aid not in emb_dict: continue
            self._init_arm(aid)
            x = make_context(user_features, emb_dict[aid])
            var = max((self.v ** 2) * (x @ self.A_inv[aid] @ x), 1e-6)
            score = np.random.normal(x @ self.mu[aid], np.sqrt(var))
            if score > best_score:
                best_score, best_arm, best_context = score, aid, x
        self.last_context = best_context
        self.last_arm = best_arm
        return best_arm

    def update(self, reward):
        if self.last_context is None or self.last_arm is None: return
        x = self.last_context
        aid = self.last_arm
        Ax = self.A_inv[aid] @ x
        self.A_inv[aid] -= np.outer(Ax, Ax) / (1 + x @ Ax)
        self.b[aid] += reward * x
        self.mu[aid] = self.A_inv[aid] @ self.b[aid]

# ==========================================
# 6. SIMULATION ENGINE
# ==========================================
def test_algo(algo, df, emb_dict, news_to_cat):
    cumulative_reward = 0
    results = []
    total_rows = len(df)
    start_time = time.time()
    last_print_time = start_time

    print(f"\nEvaluating: {algo.name}")
    print(f"{'Step':>10} | {'Progress':>8} | {'Cum Reward':>10} | {'CTR':>8} | {'Speed':>10}")
    print("-" * 75)

    for i, row in enumerate(df.itertuples(), start=1):
        chosen = algo.select_arm(
            user_features=row.user_features, 
            candidates=row.candidate_ids, 
            emb_dict=emb_dict,
            news_to_cat=news_to_cat
        )

        if chosen is None:
            continue

        reward = 1 if chosen == row.clicked_id else 0
        algo.update(reward)
        cumulative_reward += reward

        results.append({
            "Timestep": i, 
            "Reward": reward, 
            "Cumulative_Reward": cumulative_reward
        })

        if i % 1000 == 0 or i == total_rows:
            curr = time.time()
            elapsed = curr - start_time
            speed = 1000 / (curr - last_print_time) if i > 1000 else i/elapsed
            ctr = cumulative_reward / i
            print(f"{i:10d} | {(i/total_rows)*100:7.1f}% | {cumulative_reward:10d} | {ctr:8.4f} | {speed:7.1f} s/s")
            last_print_time = curr

    return pd.DataFrame(results)

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 11 Total Algorithms
    parser.add_argument("algo", choices=[
        "eps", "decay_eps", "softmax", "ucb1", "ts",           # Non-Contextual
        "shared_eps", "shared_linucb", "shared_ts",            # Shared Contextual
        "disjoint_eps", "disjoint_linucb", "disjoint_ts"       # Disjoint Contextual
    ])
    args = parser.parse_args()
    simulations = 10
    
    df_eval = users_interaction.sample(frac=1, random_state=42).reset_index(drop=True)
    context_dim = len(df_eval.iloc[0]["user_features"]) + 64

    # Map string arguments to instances
    algo_map = {
        "eps": lambda: EpsilonGreedy(),
        "decay_eps": lambda: DecayingEpsilon(),
        "softmax": lambda: Softmax(),
        "ucb1": lambda: UCB1(),
        "ts": lambda: ThompsonSampling(),
        "shared_eps": lambda: SharedEpsilonGreedy(context_dim),
        "shared_linucb": lambda: SharedLinUCB(context_dim),
        "shared_ts": lambda: SharedTS(context_dim),
        "disjoint_eps": lambda: DisjointEpsilonGreedy(context_dim),
        "disjoint_linucb": lambda: DisjointLinUCB(context_dim),
        "disjoint_ts": lambda: DisjointTS(context_dim)
    }

    AlgoFactory = algo_map[args.algo]

    # Deterministic algorithms don't need multiple simulations
    if args.algo in ["ucb1", "shared_linucb", "disjoint_linucb"]:
        results = test_algo(AlgoFactory(), df_eval, article_embedding_dict_64, news_to_category)
        results["Simulation"] = 0 
    else:
        all_runs = []
        for sim in range(simulations):
            res = test_algo(AlgoFactory(), df_eval, article_embedding_dict_64, news_to_category)
            res["Simulation"] = sim
            all_runs.append(res)
        results = pd.concat(all_runs)

    results.to_csv(f"results_{args.algo}.csv", index=False)
    print(f"\nDone. Results saved to results_{args.algo}.csv")