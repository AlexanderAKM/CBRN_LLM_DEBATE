#!/usr/bin/env python3
"""
Minority Persuasion Analysis
===========================
Analyzes which models were most successful at convincing others when they started 
in the minority position in CBRN debate results.
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def load_all_results(results_dir: str = "results_exp1") -> List[Dict]:
    """Load all JSON result files from the results directory."""
    results_path = Path(results_dir)
    results = []
    
    for json_file in sorted(results_path.glob("debate_results_*.json")):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    print(f"Loaded {len(results)} result files")
    return results


def identify_minority_positions(individual_answers: Dict[str, str]) -> Tuple[Dict[str, List[str]], str]:
    """
    Identify which models started in minority vs majority positions.
    
    Returns:
        tuple: (position_groups, majority_answer) where position_groups contains:
               'minority': list of models in minority position
               'majority': list of models in majority position
    """
    answers = list(individual_answers.values())
    answer_counts = Counter(answers)
    
    # Find majority answer (most common)
    majority_answer = answer_counts.most_common(1)[0][0]
    majority_count = answer_counts[majority_answer]
    
    position_groups = {'minority': [], 'majority': []}
    
    for model, answer in individual_answers.items():
        if answer == majority_answer:
            position_groups['majority'].append(model)
        else:
            position_groups['minority'].append(model)
    
    return position_groups, majority_answer


def analyze_persuasion_success(debate_result: Dict) -> Dict:
    """
    Analyze if any minority position models successfully convinced others.
    
    Returns dict with persuasion success information.
    """
    individual_answers = debate_result['individual_answers']
    collective_answer = debate_result['collective_answer'].upper()
    
    # Get initial positions
    position_groups, majority_answer = identify_minority_positions(individual_answers)
    
    success_info = {
        'question': debate_result['question'][:100] + "...",
        'initial_majority_answer': majority_answer,
        'final_collective_answer': collective_answer,
        'minority_models': position_groups['minority'],
        'majority_models': position_groups['majority'],
        'persuasion_occurred': False,
        'successful_persuader': None,
        'minority_answers': {}
    }
    
    # Track what each minority model initially answered
    for model in position_groups['minority']:
        success_info['minority_answers'][model] = individual_answers[model]
    
    # Check if final collective answer matches any minority position
    for model in position_groups['minority']:
        if individual_answers[model].upper() == collective_answer:
            success_info['persuasion_occurred'] = True
            success_info['successful_persuader'] = model
            break
    
    return success_info


def analyze_all_experiments(results_list: List[Dict]) -> pd.DataFrame:
    """Analyze persuasion success across all experiments."""
    all_analyses = []
    
    for exp_idx, experiment in enumerate(results_list):
        seed = experiment['config']['seed']
        
        for question_result in experiment['results']:
            analysis = analyze_persuasion_success(question_result)
            analysis['experiment_idx'] = exp_idx
            analysis['seed'] = seed
            all_analyses.append(analysis)
    
    return pd.DataFrame(all_analyses)


def count_persuasion_successes(df: pd.DataFrame) -> pd.DataFrame:
    """Count persuasion successes by model across all experiments."""
    
    # Only look at cases where persuasion actually occurred
    successful_persuasions = df[df['persuasion_occurred'] == True]
    
    # Count successes by model
    success_counts = defaultdict(int)
    total_minority_opportunities = defaultdict(int)
    
    # Count total opportunities (times each model was in minority)
    for _, row in df.iterrows():
        for model in row['minority_models']:
            total_minority_opportunities[model] += 1
            
            # Count if this model was the successful persuader
            if row['successful_persuader'] == model:
                success_counts[model] += 1
    
    # Create summary dataframe
    summary_data = []
    for model in total_minority_opportunities.keys():
        clean_model_name = model.split('/')[-1] if '/' in model else model
        summary_data.append({
            'model': clean_model_name,
            'full_model_name': model,
            'successful_persuasions': success_counts[model],
            'minority_opportunities': total_minority_opportunities[model],
            'success_rate': success_counts[model] / total_minority_opportunities[model] if total_minority_opportunities[model] > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('successful_persuasions', ascending=False)
    
    return summary_df, successful_persuasions


def print_detailed_analysis(summary_df: pd.DataFrame, successful_persuasions: pd.DataFrame):
    """Print detailed analysis results."""
    
    print("\n" + "="*80)
    print("MINORITY PERSUASION ANALYSIS - DETAILED RESULTS")
    print("="*80)
    
    print(f"\nTotal experiments analyzed: {len(successful_persuasions['seed'].unique())} seeds")
    print(f"Total questions with minority persuasion success: {len(successful_persuasions)}")
    
    print(f"\n{'Model':<35} {'Successes':<12} {'Opportunities':<15} {'Success Rate':<12}")
    print("-" * 80)
    
    for _, row in summary_df.iterrows():
        success_rate_pct = row['success_rate'] * 100
        print(f"{row['model']:<35} {row['successful_persuasions']:<12} {row['minority_opportunities']:<15} {success_rate_pct:.1f}%")
    
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print("="*80)
    
    total_successes = summary_df['successful_persuasions'].sum()
    total_opportunities = summary_df['minority_opportunities'].sum()
    overall_success_rate = total_successes / total_opportunities if total_opportunities > 0 else 0
    
    print(f"Total minority persuasion successes: {total_successes}")
    print(f"Total minority opportunities: {total_opportunities}")
    print(f"Overall success rate: {overall_success_rate:.1%}")
    
    # Show top performers
    top_by_count = summary_df.iloc[0] if len(summary_df) > 0 else None
    top_by_rate = summary_df.sort_values('success_rate', ascending=False).iloc[0] if len(summary_df) > 0 else None
    
    if top_by_count is not None:
        print(f"\nMost successful by total count: {top_by_count['model']} ({top_by_count['successful_persuasions']} successes)")
    
    if top_by_rate is not None:
        print(f"Highest success rate: {top_by_rate['model']} ({top_by_rate['success_rate']:.1%})")


def print_seed_breakdown(successful_persuasions: pd.DataFrame):
    """Print breakdown by seed/experiment."""
    
    print(f"\n{'='*80}")
    print("BREAKDOWN BY SEED/EXPERIMENT")
    print("="*80)
    
    for seed in sorted(successful_persuasions['seed'].unique()):
        seed_data = successful_persuasions[successful_persuasions['seed'] == seed]
        seed_counts = seed_data['successful_persuader'].value_counts()
        
        print(f"\nSeed {seed}: {len(seed_data)} successful persuasions")
        for model, count in seed_counts.items():
            clean_name = model.split('/')[-1] if '/' in model else model
            print(f"  {clean_name}: {count}")


def main():
    """Main analysis function."""
    print("Analyzing Minority Persuasion Success in CBRN Debates")
    print("=" * 60)
    
    # Load all experiment results
    results = load_all_results()
    if not results:
        print("No result files found in results_exp1/ directory")
        return
    
    # Analyze all experiments
    print("Analyzing persuasion patterns...")
    analysis_df = analyze_all_experiments(results)
    
    # Count successes
    summary_df, successful_persuasions = count_persuasion_successes(analysis_df)
    
    # Print results
    print_detailed_analysis(summary_df, successful_persuasions)
    print_seed_breakdown(successful_persuasions)
    
    # Save results to CSV for further analysis
    summary_df.to_csv('minority_persuasion_summary.csv', index=False)
    successful_persuasions.to_csv('successful_persuasions_details.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"  - minority_persuasion_summary.csv")
    print(f"  - successful_persuasions_details.csv")


if __name__ == "__main__":
    main()
