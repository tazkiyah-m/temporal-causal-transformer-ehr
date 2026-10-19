import json
import numpy as np
import scipy.stats as stats

def run_tests():
    with open('g:/tazz/checkpoints/real_benchmark_results.json', 'r') as f:
        data = json.load(f)
        
    c_aurocs = [r['auroc'] for r in data['raw_runs']['Temporal Causal-Masked Transformer']]
    gru_aurocs = [r['auroc'] for r in data['raw_runs']['GRU Baseline']]
    std_aurocs = [r['auroc'] for r in data['raw_runs']['Standard Transformer']]
    xgb_aurocs = [r['auroc'] for r in data['raw_runs']['XGBoost']]
    lr_aurocs = [r['auroc'] for r in data['raw_runs']['Logistic Regression']]
    
    print("=" * 60)
    print("PAIRED TWO-TAILED T-TESTS (N = 5 SEEDS, df = 4)")
    print("Note: Uncorrected for multiple comparisons. Low statistical power due to small N.")
    print("=" * 60)
    
    for name, baseline in [
        ('GRU Baseline', gru_aurocs),
        ('Standard Transformer', std_aurocs),
        ('XGBoost', xgb_aurocs),
        ('Logistic Regression', lr_aurocs)
    ]:
        diff = np.array(c_aurocs) - np.array(baseline)
        t_stat, p_val = stats.ttest_rel(c_aurocs, baseline)
        print(f"Temporal Causal TF vs {name:20s}:")
        print(f"  Mean diff: {np.mean(diff):+.4f} | t-statistic: {t_stat:.3f} | p-value: {p_val:.4f}")
        
if __name__ == '__main__':
    run_tests()
