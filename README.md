# BT4014 Project: Article Recommendation on MIND to maximise user clicks
This project implements and evaluates multiple multi-armed bandit algorithms for personalized news recommendation using the MIND dataset.

The goal is to simulate how a recommendation system dynamically learns user preferences and selects articles that maximize user engagement (clicks) over time.

We compare both non-contextual and contextual bandit approaches, highlighting the benefits of incorporating user and article features into decision-making.

## Algorithms Implemented
### Non-Contextual Bandits
- Epsilon-Greedy  
- Decaying Epsilon-Greedy  
- Softmax (Boltzmann Exploration)  
- UCB1  
- Bayesian UCB  
- Thompson Sampling  

### Contextual Bandits
- Shared Epsilon-Greedy  
- Shared LinUCB  
- Shared Thompson Sampling  
- Disjoint Epsilon-Greedy  
- Disjoint LinUCB  
- Disjoint Thompson Sampling  

## Dataset
We use the Microsoft MIND (News Recommendation Dataset).

- news.tsv: Article metadata
- behaviors.tsv: User interactions (impressions & clicks)
- article_embeddings.npy: Article embeddings generated from "all-MiniLM-L6-v2"

## Jyupter Notebook & Python File
1. BT4014_Exploratory.ipynb: Performs **descriptive analytics** on the dataset, including data exploration, distributions, and initial insights.

2. BT4014_Plots.ipynb:  Generates **visualisations of simulation results**, such as:
  - Average cumulative reward
  - Average click-through rate

3. simulation.py: Core script that runs the **bandit algorithms**

## Instructions
