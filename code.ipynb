# ============================================
# MODEL COMPARISON FUNCTION
# ============================================

def compare_models(behaviors_df, news_to_category, all_categories, n_simulations=3):
    """
    Compare different bandit algorithms
    """
    results = []
    
    # Define algorithms to compare
    algorithms = [
        ('Random', RandomBandit),
        ('EpsilonGreedy', EpsilonGreedy),
        ('DecayingEpsilon', DecayingEpsilonGreedy),
        ('LinUCB', LinUCBBandit),
        ('ThompsonSampling', ThompsonSamplingBandit)
    ]
    
    for algo_name, AlgoClass in algorithms:
        print(f"Running {algo_name}...")
        
        for sim in range(n_simulations):
            bandit = AlgoClass(len(all_categories), n_features)
            cumulative_reward = 0
            timestep = 0
            
            for idx, row in behaviors_df.iterrows():
                # Get context
                context = get_complete_context(row['History'], row['Time_Stamp'], 
                                               all_categories, news_to_category)
                
                # Get impressions
                impressions = get_impression_categories(row['Impression'], news_to_category)
                
                if len(impressions) == 0:
                    continue
                
                # Select category
                selected_cat_idx = bandit.select_category(context)
                selected_category = all_categories[selected_cat_idx]
                
                # Get reward
                reward = 0
                for cat, click in impressions:
                    if cat == selected_category:
                        reward = click
                        break
                
                # Update
                bandit.update(selected_cat_idx, context, reward)
                
                cumulative_reward += reward
                timestep += 1
                
                results.append({
                    'Algorithm': algo_name,
                    'Simulation': sim + 1,
                    'Timestep': timestep,
                    'Reward': reward,
                    'Cumulative_Reward': cumulative_reward
                })
    
    return pd.DataFrame(results)


# ============================================
# PLOT COMPARISON
# ============================================

def plot_comparison(results_df):
    """
    Plot cumulative reward comparison across algorithms
    """
    plt.figure(figsize=(12, 8))
    
    for algo in results_df['Algorithm'].unique():
        algo_data = results_df[results_df['Algorithm'] == algo]
        avg_cumulative = algo_data.groupby('Timestep')['Cumulative_Reward'].mean()
        plt.plot(avg_cumulative.index, avg_cumulative.values, label=algo, linewidth=2)
    
    plt.xlabel('Timestep')
    plt.ylabel('Average Cumulative Reward')
    plt.title('Model Comparison: Cumulative Reward Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_ctr_comparison(results_df):
    """
    Plot average CTR comparison across algorithms
    """
    avg_ctr = results_df.groupby('Algorithm')['Reward'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#95a5a6']
    bars = plt.bar(avg_ctr.index, avg_ctr.values, color=colors)
    
    plt.xlabel('Algorithm')
    plt.ylabel('Average Click-Through Rate (CTR)')
    plt.title('Model Comparison: Average CTR')
    plt.ylim(0, max(avg_ctr.values) * 1.1)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_ctr.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return avg_ctr


# ============================================
# PRINT SUMMARY TABLE
# ============================================

def print_summary(results_df):
    """
    Print a formatted summary table of results
    """
    summary = results_df.groupby('Algorithm').agg({
        'Reward': ['mean', 'std'],
        'Cumulative_Reward': 'max'
    }).round(4)
    
    summary.columns = ['Avg CTR', 'CTR Std', 'Total Reward']
    summary = summary.sort_values('Avg CTR', ascending=False)
    
    print("\n" + "="*50)
    print("MODEL COMPARISON SUMMARY")
    print("="*50)
    print(summary.to_string())
    print("="*50)
    
    return summary
