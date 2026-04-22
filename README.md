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
The results of the simulations are already available in the `/results` directory. You can directly run the `BT4014_Plots.ipynb` notebook to analyze and visualize these existing results.

### Option 1: Using GitHub Actions (Remote)
If you wish to re-run the algorithm and generate fresh results using GitHub Actions:
1. Go to the GitHub Actions page: [https://github.com/eduymil/BT4014/actions](https://github.com/eduymil/BT4014/actions)
2. Download the generated CSV results from the latest successful workflow run.
3. Place the downloaded CSV files into the local `/results` directory.
4. Open and run the `BT4014_Plots.ipynb` notebook to perform the analysis and view the plots based on the new data.

### Option 2: Running Locally
If you prefer to run the simulation locally:
1. Ensure the `dataset` folder is present in your project directory containing: `news.tsv`, `behaviors.tsv`, and `article_embeddings.npy`.
2. Open a terminal in the project directory.
3. Execute the script using Python and provide an algorithm name:
   ```bash
   python run_simulation.py <algo_name>
   ```
   *Available Algorithms:* `eps`, `decay_eps`, `softmax`, `ucb1`, `bayes_ucb`, `ts`, `shared_eps`, `shared_linucb`, `shared_ts`, `disjoint_eps`, `disjoint_linucb`, `disjoint_ts`
4. The script will generate a new `results_<algo_name>.csv` file. To analyze new findings, move this file to `/results` and run the `BT4014_Plots.ipynb` notebook.
