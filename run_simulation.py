import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
data_path_news = Path("dataset") / "news.tsv"
data_path_user_interaction = Path("dataset") / "behaviors.tsv"
data_path_news_embeddings = Path("dataset") / "article_embeddings.npy"

news_articles = pd.read_csv(data_path_news, sep="\t", header=None)
news_articles.columns = ['News_ID', 'Category', 'Subcategory', 'Title', 'Abstract', 'URL', 'Title Entities', 'Abstract Entities']

users_interaction = pd.read_csv(data_path_user_interaction, sep="\t", header=None)
users_interaction.columns = ['No.', 'User_ID', 'Time_Stamp', 'History', 'Impression']

article_embeddings = np.load(data_path_news_embeddings)
article_embedding_dict = {
    news_id: emb for news_id, emb in zip(news_articles["News_ID"], article_embeddings)
}

# History count feature
def get_history_count(history):
    if pd.isna(history) or history == "": return 0
    return len(history.split())

users_interaction['history_count'] = users_interaction['History'].apply(get_history_count)
users_interaction['history_count_norm'] = users_interaction['history_count'] / users_interaction['history_count'].max()

# Category Frequency
news_to_category = dict(zip(news_articles['News_ID'], news_articles['Category']))
all_categories = news_articles['Category'].unique().tolist()

def get_category_freq(history):
    freq = defaultdict(int)
    if pd.isna(history) or history == "": return np.zeros(len(all_categories))
    for article in history.split():
        category = news_to_category.get(article)
        if category: freq[category] += 1
    return np.array([freq[cat] for cat in all_categories])

users_interaction['category_freq'] = users_interaction['History'].apply(get_category_freq)

# Time Features
users_interaction['Time_Stamp'] = pd.to_datetime(users_interaction['Time_Stamp'], format='%m/%d/%Y %I:%M:%S %p')
users_interaction['hour'] = pd.to_datetime(users_interaction['Time_Stamp']).dt.hour
users_interaction['dayofweek'] = pd.to_datetime(users_interaction['Time_Stamp']).dt.dayofweek
users_interaction['hour_norm'] = users_interaction['hour'] / 23.0
users_interaction['day_norm'] = users_interaction['dayofweek'] / 6.0

# User Vector
users_interaction['history_list'] = users_interaction['History'].fillna('').apply(lambda x: x.split())
def get_user_vector(history):
    vectors = [article_embedding_dict[a] for a in history if a in article_embedding_dict]
    if len(vectors) == 0: return np.zeros(384)
    return np.mean(vectors, axis=0)
users_interaction['user_vector'] = users_interaction['history_list'].apply(get_user_vector)

def build_user_features(row):
    time_features = np.array([row['hour_norm'], row['day_norm']])
    frequency_features = np.array([row['history_count_norm']])
    return np.concatenate([row['user_vector'], time_features, frequency_features, row['category_freq']])

users_interaction['user_features'] = users_interaction.apply(build_user_features, axis=1)

# PCA Reduction
article_ids = list(article_embedding_dict.keys())
X_articles = np.array([article_embedding_dict[aid] for aid in article_ids])
pca_64 = PCA(n_components=64, random_state=42)
X_articles_64 = pca_64.fit_transform(X_articles)
article_embedding_dict_64 = {aid: X_articles_64[i] for i, aid in enumerate(article_ids)}

X_users = np.array(users_interaction["user_vector"].tolist())
X_users_64 = pca_64.transform(X_users)
users_interaction["user_vector_64"] = list(X_users_64)

# ==========================================
# OPTIMIZATION: Pre-parse strings ONCE
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
# 2. ALGORITHM CLASSES
# ==========================================
class DecayingEpsilonGreedyLinear:
    def __init__(self, d, initial_epsilon=1.0, decay_rate=0.001, lr=0.01):
        self.name = "Decaying Epsilon Greedy"
        self.d = d
        self.theta = np.zeros(d)
        self.initial_epsilon = initial_epsilon
        self.decay_rate = decay_rate
        self.lr = lr
        self.t = 1
        self.last_x = None

    def get_epsilon(self):
        return self.initial_epsilon / (1 + self.decay_rate * self.t)

    def predict(self, x):
        return np.dot(self.theta, x)

    def select_arm(self, user_features, candidate_article_ids, article_embedding_dict):
        epsilon = self.get_epsilon()
        valid_candidates = [aid for aid in candidate_article_ids if aid in article_embedding_dict]
        if not valid_candidates:
            self.last_x = None
            return None

        if np.random.rand() < epsilon:
            chosen = np.random.choice(valid_candidates)
            self.last_x = make_context(user_features, article_embedding_dict[chosen])
            return chosen

        best_score, best_article, best_x = -np.inf, None, None
        for aid in valid_candidates:
            x = make_context(user_features, article_embedding_dict[aid])
            score = self.predict(x)
            if score > best_score:
                best_score, best_article, best_x = score, aid, x
        self.last_x = best_x
        return best_article

    def update(self, chosen_article_id, reward):
        if self.last_x is None: return
        error = reward - self.predict(self.last_x)
        self.theta += self.lr * error * self.last_x
        self.t += 1

class LinUCB:
    def __init__(self, d, alpha=1.0):
        self.name = "LinUCB"
        self.d = d
        self.alpha = alpha
        self.A_inv = {}
        self.b = {}
        self.theta = {}
        self.last_context = {}

    def _init_arm(self, article_id):
        if article_id not in self.A_inv:
            self.A_inv[article_id] = np.eye(self.d)
            self.b[article_id] = np.zeros(self.d)
            self.theta[article_id] = np.zeros(self.d)

    def select_arm(self, user_features, candidate_article_ids, article_embedding_dict):
        self.last_context = {}
        best_score, best_article = -np.inf, None

        for aid in candidate_article_ids:
            self._init_arm(aid)
            A_inv_a, theta_a = self.A_inv[aid], self.theta[aid]
            x = make_context(user_features, article_embedding_dict[aid])
            self.last_context[aid] = x

            mean_reward = x @ theta_a
            uncertainty = np.sqrt(x @ A_inv_a @ x)
            score = mean_reward + self.alpha * uncertainty

            if score > best_score:
                best_score, best_article = score, aid
        return best_article

    def update(self, chosen_article_id, reward):
        self._init_arm(chosen_article_id)
        x = self.last_context[chosen_article_id]
        A_inv = self.A_inv[chosen_article_id]

        Ax = A_inv @ x
        numerator = np.outer(Ax, Ax)
        denominator = 1.0 + (x @ Ax)

        self.A_inv[chosen_article_id] -= numerator / denominator
        self.b[chosen_article_id] += reward * x
        self.theta[chosen_article_id] = self.A_inv[chosen_article_id] @ self.b[chosen_article_id]

class LinearThompsonSampling:
    def __init__(self, d, lambda_=1.0, v=0.3):
        self.name = "Linear Thompson Sampling"
        self.d = d
        self.lambda_ = lambda_
        self.v = v
        self.A_inv = {}
        self.b = {}
        self.mu = {}
        self.last_context = {}

    def _init_arm(self, aid):
        if aid not in self.A_inv:
            self.A_inv[aid] = (1.0 / self.lambda_) * np.eye(self.d)
            self.b[aid] = np.zeros(self.d)
            self.mu[aid] = np.zeros(self.d)

    def select_arm(self, user_features, candidate_article_ids, article_embedding_dict):
        best_score, best_article = -np.inf, None
        self.last_context = {}

        for aid in candidate_article_ids:
            self._init_arm(aid)
            cov = (self.v ** 2) * self.A_inv[aid]
            theta_sample = np.random.multivariate_normal(self.mu[aid], cov)
            x = make_context(user_features, article_embedding_dict[aid])
            self.last_context[aid] = x
            
            score = x @ theta_sample
            if score > best_score:
                best_score, best_article = score, aid
        return best_article

    def update(self, chosen_article_id, reward):
        if not chosen_article_id: return
        self._init_arm(chosen_article_id)
        x = self.last_context.get(chosen_article_id)
        if x is None: return

        A_inv = self.A_inv[chosen_article_id]
        Ax = A_inv @ x
        numerator = np.outer(Ax, Ax)
        denominator = 1.0 + (x @ Ax)

        self.A_inv[chosen_article_id] -= numerator / denominator
        self.b[chosen_article_id] += reward * x
        self.mu[chosen_article_id] = self.A_inv[chosen_article_id] @ self.b[chosen_article_id]

# ==========================================
# 3. FAST SIMULATION ENGINE
# ==========================================
def test_algo(algo, interaction_df, article_embedding_dict, user_col="user_vector_64"):
    results = []
    cumulative_reward = 0.0

    for t, row in enumerate(interaction_df.itertuples(), start=1):
        user_features = getattr(row, user_col)
        candidate_article_ids = row.candidate_ids
        clicked_article_id = row.clicked_id
        
        valid_candidates = [aid for aid in candidate_article_ids if aid in article_embedding_dict]
        if not valid_candidates: continue

        chosen_article_id = algo.select_arm(user_features, valid_candidates, article_embedding_dict)
        reward = 1 if chosen_article_id == clicked_article_id else 0

        algo.update(chosen_article_id, reward)
        cumulative_reward += reward

        results.append({
            "Algorithm": algo.name,
            "Timestep": t,
            "Chosen_Arm": chosen_article_id,
            "Reward": reward,
            "Cumulative_Reward": cumulative_reward
        })
    return pd.DataFrame(results)

# ==========================================
# 4. EXECUTION HANDLER
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Contextual Bandit Algorithm")
    parser.add_argument("algo", choices=["eps", "linucb", "ts"], help="Algorithm to run")
    args = parser.parse_args()

    # Fixed evaluation subset
    users_interaction_5000 = users_interaction.iloc[:5000].copy()
    context_dim = len(users_interaction_5000.iloc[0]["user_vector_64"]) + X_articles_64.shape[1]
    
    simulation_runs = 10
    all_runs = []

    print(f"Starting execution for: {args.algo.upper()}")

    if args.algo == "eps":
        for sim in range(simulation_runs):
            algo = DecayingEpsilonGreedyLinear(d=context_dim)
            res = test_algo(algo, users_interaction_5000, article_embedding_dict_64)
            res["Simulation"] = sim + 1
            all_runs.append(res)

    elif args.algo == "linucb":
        # LinUCB is deterministic, so 1 simulation is enough
        algo = LinUCB(d=context_dim, alpha=1.0)
        res = test_algo(algo, users_interaction_5000, article_embedding_dict_64)
        res["Simulation"] = 1
        all_runs.append(res)

    elif args.algo == "ts":
        for sim in range(simulation_runs):
            algo = LinearThompsonSampling(d=context_dim, lambda_=1.0, v=0.3)
            res = test_algo(algo, users_interaction_5000, article_embedding_dict_64)
            res["Simulation"] = sim + 1
            all_runs.append(res)

    # Save to CSV
    final_df = pd.concat(all_runs, ignore_index=True)
    out_file = f"results_{args.algo}.csv"
    final_df.to_csv(out_file, index=False)
    print(f"Finished! Saved results to {out_file}")