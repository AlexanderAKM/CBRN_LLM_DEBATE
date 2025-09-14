#!/usr/bin/env python3
"""
CBRN Debate Results Analysis Script
==================================
Performs statistical significance testing and creates bar plots for debate results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd


def load_results(results_dir: str = "results") -> List[Dict]:
    """Load all JSON result files from the results directory."""
    results_path = Path(results_dir)
    results = []
    
    for json_file in results_path.glob("debate_results_*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    print(f"Loaded {len(results)} result files")
    return results


def extract_scores(results: List[Dict]) -> Tuple[pd.DataFrame, List[float]]:
    """Extract individual and collective scores from results."""
    individual_scores = []
    collective_scores = []
    
    for result in results:
        # Extract individual model scores
        individual = result['safety_scores']['individual']
        for model, score in individual.items():
            individual_scores.append({
                'model': model.split('/')[-1],  # Clean model name
                'score': score,
                'experiment': result['timestamp']
            })
        
        # Extract collective score
        collective_scores.append(result['safety_scores']['collective'])
    
    individual_df = pd.DataFrame(individual_scores)
    return individual_df, collective_scores


def perform_significance_tests(individual_df: pd.DataFrame, collective_scores: List[float]) -> Dict:
    """Perform statistical significance tests."""
    results = {}
    
    # Test if collective performance is significantly better than individual
    all_individual_scores = individual_df['score'].values
    
    # Paired t-test (comparing collective vs average individual per experiment)
    individual_means = individual_df.groupby('experiment')['score'].mean().values
    t_stat, p_value = stats.ttest_rel(collective_scores, individual_means)
    
    results['collective_vs_individual'] = {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': (np.mean(collective_scores) - np.mean(individual_means)) / np.std(individual_means)
    }
    
    # ANOVA across individual models
    model_groups = [group['score'].values for name, group in individual_df.groupby('model')]
    f_stat, p_value_anova = stats.f_oneway(*model_groups)
    
    results['anova_individual_models'] = {
        'f_statistic': f_stat,
        'p_value': p_value_anova,
        'significant': p_value_anova < 0.05
    }
    
    # Post-hoc pairwise tests if ANOVA is significant
    if p_value_anova < 0.05:
        model_names = individual_df['model'].unique()
        pairwise_results = {}
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                scores1 = individual_df[individual_df['model'] == model1]['score'].values
                scores2 = individual_df[individual_df['model'] == model2]['score'].values
                t_stat, p_val = stats.ttest_ind(scores1, scores2)
                
                pairwise_results[f"{model1}_vs_{model2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05
                }
        
        results['pairwise_comparisons'] = pairwise_results
    
    return results


def create_bar_plot(individual_df: pd.DataFrame, collective_scores: List[float], save_path: str = "results_plot.png"):
    """Create a clean bar plot comparing individual and collective performance."""
    plt.style.use('seaborn-v0_8')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate means and standard errors
    individual_stats = individual_df.groupby('model')['score'].agg(['mean', 'std', 'count']).reset_index()
    individual_stats['se'] = individual_stats['std'] / np.sqrt(individual_stats['count'])
    
    collective_mean = np.mean(collective_scores)
    collective_se = np.std(collective_scores) / np.sqrt(len(collective_scores))
    
    # Create bar plot
    x_pos = np.arange(len(individual_stats) + 1)
    
    # Individual model bars
    bars1 = ax.bar(x_pos[:-1], individual_stats['mean'], 
                   yerr=individual_stats['se'], 
                   capsize=5, alpha=0.8, color='lightblue', 
                   label='Individual Models')
    
    # Collective bar
    bars2 = ax.bar(x_pos[-1], collective_mean, 
                   yerr=collective_se, 
                   capsize=5, alpha=0.8, color='orange', 
                   label='Collective')
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Safety Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('CBRN Safety Assessment: Individual vs Collective Performance', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels
    labels = [name.replace('-', '-\n') for name in individual_stats['model']] + ['Collective']
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + individual_stats.iloc[i]['se'] + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.text(bars2[0].get_x() + bars2[0].get_width()/2., collective_mean + collective_se + 0.5,
            f'{collective_mean:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {save_path}")


def print_results_summary(stats_results: Dict, individual_df: pd.DataFrame, collective_scores: List[float]):
    """Print a clean summary of the statistical results."""
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*60)
    
    # Basic descriptives
    print(f"\nDESCRIPTIVE STATISTICS:")
    print(f"Number of experiments: {len(collective_scores)}")
    print(f"Collective mean: {np.mean(collective_scores):.2f}% (SD: {np.std(collective_scores):.2f})")
    
    individual_summary = individual_df.groupby('model')['score'].agg(['mean', 'std']).round(2)
    print(f"\nIndividual model performance:")
    for model, row in individual_summary.iterrows():
        print(f"  {model}: {row['mean']:.2f}% (SD: {row['std']:.2f})")
    
    # Significance tests
    print(f"\nSIGNIFICANCE TESTS:")
    
    collective_test = stats_results['collective_vs_individual']
    print(f"Collective vs Individual (paired t-test):")
    print(f"  t = {collective_test['t_statistic']:.3f}, p = {collective_test['p_value']:.4f}")
    print(f"  Effect size (Cohen's d): {collective_test['effect_size']:.3f}")
    print(f"  Significant: {'YES' if collective_test['significant'] else 'NO'}")
    
    anova_test = stats_results['anova_individual_models']
    print(f"\nIndividual models comparison (ANOVA):")
    print(f"  F = {anova_test['f_statistic']:.3f}, p = {anova_test['p_value']:.4f}")
    print(f"  Significant: {'YES' if anova_test['significant'] else 'NO'}")
    
    if 'pairwise_comparisons' in stats_results:
        print(f"\nPairwise comparisons (significant only):")
        for comparison, result in stats_results['pairwise_comparisons'].items():
            if result['significant']:
                print(f"  {comparison}: p = {result['p_value']:.4f}")


def main():
    """Main analysis function."""
    print("CBRN Debate Results Analysis")
    print("=" * 40)
    
    # Load data
    results = load_results()
    if not results:
        print("No result files found in results/ directory")
        return
    
    # Extract scores
    individual_df, collective_scores = extract_scores(results)
    
    # Perform statistical tests
    stats_results = perform_significance_tests(individual_df, collective_scores)
    
    # Create visualization
    create_bar_plot(individual_df, collective_scores)
    
    # Print summary
    print_results_summary(stats_results, individual_df, collective_scores)


if __name__ == "__main__":
    main()
