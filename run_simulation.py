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

# ==========================================
# 2. FEATURE ENGINEERING (OPTIMIZED)
# ==========================================
print("Feature engineering...")

# 2.1 History Count
def get_history_count(history):
    if pd.isna(history) or history == "": return 0
    return len(history.split())

users_interaction['history_count'] = users_interaction['History'].apply(get_history_count)
users_interaction['history_count_norm'] = users_interaction['history_count'] / users_interaction['history_count'].max()

# 2.2 Category Frequencies
news_to_category = dict(zip(news_articles['News_ID'], news_articles['Category']))
all_categories = news_articles['Category'].unique().tolist()

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

# 2.3 Time Features
users_interaction['Time_Stamp'] = pd.to_datetime(users_interaction['Time_Stamp'])
users_interaction['hour_norm'] = users_interaction['Time_Stamp'].dt.hour / 23.0
users_interaction['day_norm'] = users_interaction['Time_Stamp'].dt.dayofweek / 6.0

# 2.4 Mean User Embeddings
def get_user_vector(history):
    vecs = [article_embedding_dict[a] for a in history.split() if a in article_embedding_dict]
    if len(vecs) == 0: return np.zeros(EMB_DIM)
    return np.mean(vecs, axis=0)

users_interaction['user_vector'] = users_interaction['History'].fillna('').apply(get_user_vector)

# ==========================================
# 3. PCA & VECTORIZED FEATURE BUILDING
# ==========================================
print("Running PCA & Vectorizing User Features...")

# PCA on articles
article_ids = list(article_embedding_dict.keys())
X_articles = np.array([article_embedding_dict[aid] for aid in article_ids])
pca = PCA(n_components=64, random_state=42)
X_articles_64 = pca.fit_transform(X_articles)
article_embedding_dict_64 = {aid: X_articles_64[i] for i, aid in enumerate(article_ids)}

# Vectorized User Feature Construction
X_users_pca = pca.transform(np.vstack(users_interaction["user_vector"].values))
time_feats = users_interaction[["hour_norm", "day_norm"]].values
hist_feat = users_interaction[["history_count_norm"]].values
cat_feats = np.vstack(users_interaction["category_freq"].values)

# Final combined matrix
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
# 5. BANDIT ALGORITHMS (SHARED SPACE)
# ==========================================

class SharedEpsilonGreedy:
    def __init__(self, d, epsilon=0.1, lr=0.01):
        self.name = "Epsilon Greedy"
        self.theta = np.zeros(d)
        self.epsilon = epsilon
        self.lr = lr
        self.last_context = None

    def select_arm(self, user_features, candidates, emb_dict):
        valid = [aid for aid in candidates if aid in emb_dict]
        if not valid: return None
        
        if np.random.rand() < self.epsilon:
            chosen = np.random.choice(valid)
            self.last_context = make_context(user_features, emb_dict[chosen])
            return chosen

        scores = []
        contexts = []
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
        self.name = "LinUCB"
        self.alpha = alpha
        self.A_inv = np.eye(d)
        self.b = np.zeros(d)
        self.theta = np.zeros(d)
        self.last_context = None

    def select_arm(self, user_features, candidates, emb_dict):
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
        self.name = "Thompson Sampling"
        self.A_inv = np.eye(d)
        self.b = np.zeros(d)
        self.mu = np.zeros(d)
        self.v = v
        self.last_context = None

    def select_arm(self, user_features, candidates, emb_dict):
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
# 6. SIMULATION ENGINE
# ==========================================
def test_algo(algo, df, emb_dict):
    cumulative_reward, t_valid = 0, 0
    results = []
    total_rows = len(df)
    start_time = time.time()
    last_print_time = start_time

    print(f"\nEvaluating: {algo.name}")
    print(f"{'Step':>10} | {'Progress':>8} | {'Accepted':>10} | {'CTR':>8} | {'Match%':>8} | {'Speed':>10}")
    print("-" * 80)

    for i, row in enumerate(df.itertuples(), start=1):
        chosen = algo.select_arm(row.user_features, row.candidate_ids, emb_dict)

        # Replay Match Logic
        if chosen == row.clicked_id:
            reward = 1
            algo.update(reward)
            cumulative_reward += reward
            t_valid += 1
            results.append({"Timestep": t_valid, "Reward": reward, "Cumulative_Reward": cumulative_reward})

        if i % 1000 == 0 or i == total_rows:
            curr = time.time()
            elapsed = curr - start_time
            speed = 1000 / (curr - last_print_time) if i > 1000 else i/elapsed
            print(f"{i:10d} | {(i/total_rows)*100:7.1f}% | {t_valid:10d} | {cumulative_reward/max(t_valid,1):8.4f} | {(t_valid/i)*100:7.2f}% | {speed:7.1f} s/s")
            last_print_time = curr

    return pd.DataFrame(results)

# ==========================================
# 7. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("algo", choices=["eps", "linucb", "ts"])
    args = parser.parse_args()
    simulations = 10
    # Fixed subset for simulation
    df_eval = users_interaction.sample(frac=1, random_state=42).reset_index(drop=True)
    context_dim = len(df_eval.iloc[0]["user_features"]) + 64

    if args.algo == "eps":
        all_runs = []
        for sim in range(simulations):
            res = test_algo(SharedEpsilonGreedy(context_dim), df_eval, article_embedding_dict_64)
            res["run"] = sim
            all_runs.append(res)
        results = pd.concat(all_runs)
    
    elif args.algo == "linucb":
        results = test_algo(SharedLinUCB(context_dim), df_eval, article_embedding_dict_64)
    
    elif args.algo == "ts":
        all_runs = []
        for sim in range(simulations):
            res = test_algo(SharedTS(context_dim), df_eval, article_embedding_dict_64)
            res["run"] = sim
            all_runs.append(res)
        results = pd.concat(all_runs)

    results.to_csv(f"results_{args.algo}.csv", index=False)
    print(f"\nDone. Results saved to results_{args.algo}.csv")