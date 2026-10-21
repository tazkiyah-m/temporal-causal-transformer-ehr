import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, roc_curve, precision_recall_curve

from dataset import get_train_val_test_datasets
from model import TemporalCausalTransformer, StandardTransformer, GRUBaseline

SEEDS = [42, 43, 44, 45, 46]
EPOCHS = 8
BATCH_SIZE = 512
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_flat_features(dataset):
    """Flatten sequence (N, 10, 9) -> (N, 90) for tabular models"""
    X = dataset.samples_features.reshape(len(dataset), -1)
    y = dataset.samples_readm
    return X, y

def evaluate_predictions(y_true, y_prob):
    auroc = float(roc_auc_score(y_true, y_prob))
    auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    # Best F1 across thresholds
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_f1 = float(np.max(f1_scores))
    return {
        'auroc': round(auroc, 4),
        'auprc': round(auprc, 4),
        'brier': round(brier, 4),
        'f1': round(best_f1, 4)
    }

def train_torch_model(model, train_loader, val_loader, test_loader, epochs=10, lr=1e-3, weight_decay=1e-4):
    model = model.to(DEVICE)
    # Use pos_weight to handle 5.19% minority class
    pos_weight = torch.tensor([4.0]).to(DEVICE) # mild class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            x = batch['features'].to(DEVICE)
            y = batch['readmission_label'].float().to(DEVICE)
            
            optimizer.zero_grad()
            out = model(x)
            logits = out['logits'] # single scalar logit (B,)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * len(y)
            n_train += len(y)
            
        scheduler.step()
        train_loss /= n_train
        
        # Validation
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch['features'].to(DEVICE)
                y = batch['readmission_label'].float().to(DEVICE)
                out = model(x)
                logits = out['logits']
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                n_val += len(y)
        val_loss /= n_val
        
        history['train_loss'].append(round(train_loss, 4))
        history['val_loss'].append(round(val_loss, 4))
        
    # Evaluate on Test Set
    model.eval()
    test_probs = []
    test_targets = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch['features'].to(DEVICE)
            y = batch['readmission_label']
            out = model(x)
            probs = torch.sigmoid(out['logits']).cpu().numpy()
            test_probs.extend(probs)
            test_targets.extend(y.numpy())
            
    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    metrics = evaluate_predictions(test_targets, test_probs)
    return metrics, test_probs, history, model

def run_all_benchmarks():
    print("=" * 60)
    print("STARTING RIGOROUS MULTI-SEED EXPERIMENTS ON REAL UCI DATA")
    print(f"Seeds: {SEEDS} | Epochs: {EPOCHS} | Batch Size: {BATCH_SIZE} | Device: {DEVICE}")
    print("=" * 60)
    
    all_results = {
        'Logistic Regression': [],
        'XGBoost': [],
        'GRU Baseline': [],
        'Standard Transformer': [],
        'Temporal Causal-Masked Transformer': []
    }
    
    # Storage for detailed Seed 42 outputs (curves, patient sample, t-SNE)
    seed42_payload = {}
    
    for seed in SEEDS:
        print(f"\n---> Running Experiments for SEED {seed}...")
        train_ds, val_ds, test_ds = get_train_val_test_datasets(seed=seed)
        
        # 1. Tabular Data Preparation
        X_train, y_train = extract_flat_features(train_ds)
        X_test, y_test = extract_flat_features(test_ds)
        
        # Model 1: Logistic Regression
        lr_model = LogisticRegression(max_iter=500, class_weight='balanced', random_state=seed)
        lr_model.fit(X_train, y_train)
        lr_prob = lr_model.predict_proba(X_test)[:, 1]
        lr_metrics = evaluate_predictions(y_test, lr_prob)
        all_results['Logistic Regression'].append(lr_metrics)
        print(f"  [LR]        AUROC: {lr_metrics['auroc']} | AUPRC: {lr_metrics['auprc']} | F1: {lr_metrics['f1']}")
        
        # Model 2: XGBoost
        scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
        xgb_model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            scale_pos_weight=scale_pos, eval_metric='logloss',
            random_state=seed, n_jobs=-1
        )
        xgb_model.fit(X_train, y_train)
        xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
        xgb_metrics = evaluate_predictions(y_test, xgb_prob)
        all_results['XGBoost'].append(xgb_metrics)
        print(f"  [XGBoost]   AUROC: {xgb_metrics['auroc']} | AUPRC: {xgb_metrics['auprc']} | F1: {xgb_metrics['f1']}")
        
        # 2. PyTorch Data Loaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Model 3: GRU Baseline
        gru = GRUBaseline(input_dim=9, hidden_dim=64, num_layers=2)
        gru_metrics, gru_probs, _, _ = train_torch_model(gru, train_loader, val_loader, test_loader, epochs=EPOCHS)
        all_results['GRU Baseline'].append(gru_metrics)
        print(f"  [GRU]       AUROC: {gru_metrics['auroc']} | AUPRC: {gru_metrics['auprc']} | F1: {gru_metrics['f1']}")
        
        # Model 4: Standard Transformer (No mask)
        std_tf = StandardTransformer(input_dim=9, d_model=64, nhead=4, num_layers=2)
        std_metrics, std_probs, _, _ = train_torch_model(std_tf, train_loader, val_loader, test_loader, epochs=EPOCHS)
        all_results['Standard Transformer'].append(std_metrics)
        print(f"  [Std Trans] AUROC: {std_metrics['auroc']} | AUPRC: {std_metrics['auprc']} | F1: {std_metrics['f1']}")
        
        # Model 5: Temporal Causal-Masked Transformer
        causal_tf = TemporalCausalTransformer(input_dim=9, d_model=64, nhead=4, num_layers=2)
        c_metrics, c_probs, history, trained_c_model = train_torch_model(causal_tf, train_loader, val_loader, test_loader, epochs=EPOCHS)
        all_results['Temporal Causal-Masked Transformer'].append(c_metrics)
        print(f"  [Causal-TF] AUROC: {c_metrics['auroc']} | AUPRC: {c_metrics['auprc']} | F1: {c_metrics['f1']}")
        
        # If seed == 42, capture curves, latent representations, and real patient details
        if seed == 42:
            fpr_c, tpr_c, _ = roc_curve(y_test, c_probs)
            prec_c, rec_c, _ = precision_recall_curve(y_test, c_probs)
            fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
            fpr_gru, tpr_gru, _ = roc_curve(y_test, gru_probs)
            fpr_std, tpr_std, _ = roc_curve(y_test, std_probs)
            
            # Authentic multi-encounter patient in held-out test set (Patient #2448531, 10 genuine visits)
            real_patient_info = None
            target_pt_nbr = 2448531
            for p_idx, p_nbr in enumerate(test_ds.valid_patient_nbrs):
                if p_nbr == target_pt_nbr:
                    p_feat = test_ds.samples_features[p_idx]
                    p_label = int(test_ds.samples_readm[p_idx])
                    p_tensor = torch.tensor(p_feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                    
                    # Step-by-step risk evaluated via Right-Replication Partial History (Section III-A-3)
                    # For each visit step t (1..10), right-replicate encounter t across subsequent steps
                    step_risks = []
                    for t in range(1, 11):
                        sub_seq = p_tensor.clone()
                        for j in range(t, 10):
                            sub_seq[:, j, :] = sub_seq[:, t-1, :]
                        out = trained_c_model(sub_seq)
                        risk = float(torch.sigmoid(out['logits']).cpu().detach())
                        step_risks.append(round(risk, 4))
                    
                    real_patient_info = {
                        'patient_nbr': int(p_nbr),
                        'observed_readmitted_30d': p_label,
                        'time_in_hospital_raw': [float(round(val * 14.0, 1)) for val in p_feat[:, 1]],
                        'lab_procedures_raw': [int(round(val * 132.0)) for val in p_feat[:, 2]],
                        'medications_raw': [int(round(val * 81.0)) for val in p_feat[:, 4]],
                        'simulated_step_risks': step_risks
                    }
                    break
            
            # Missing data robustness test on test set
            robustness_results = {'missing_pcts': [0, 10, 20, 30, 40, 50], 'causal_tf': [], 'gru': []}
            for pct in [0, 10, 20, 30, 40, 50]:
                mask_rate = pct / 100.0
                np.random.seed(42)
                X_corrupt = test_ds.samples_features.copy()
                if mask_rate > 0:
                    corrupt_mask = np.random.rand(*X_corrupt.shape) < mask_rate
                    X_corrupt[corrupt_mask] = 0.0
                
                corrupt_tensor = torch.tensor(X_corrupt, dtype=torch.float32)
                with torch.no_grad():
                    # Temporal Causal-Masked Transformer
                    out_c = trained_c_model(corrupt_tensor.to(DEVICE))
                    prob_c_corrupt = torch.sigmoid(out_c['logits']).cpu().numpy()
                    robustness_results['causal_tf'].append(round(float(roc_auc_score(y_test, prob_c_corrupt)), 4))
                    
                    # GRU
                    out_g = gru(corrupt_tensor.to(DEVICE))
                    prob_g_corrupt = torch.sigmoid(out_g['logits']).cpu().numpy()
                    robustness_results['gru'].append(round(float(roc_auc_score(y_test, prob_g_corrupt)), 4))
            
            seed42_payload = {
                'history': history,
                'real_patient': real_patient_info,
                'robustness': robustness_results,
                'roc_curves': {
                    'causal_tf': {'fpr': fpr_c[::max(1, len(fpr_c)//100)].tolist(), 'tpr': tpr_c[::max(1, len(tpr_c)//100)].tolist()},
                    'xgb': {'fpr': fpr_xgb[::max(1, len(fpr_xgb)//100)].tolist(), 'tpr': tpr_xgb[::max(1, len(tpr_xgb)//100)].tolist()},
                    'gru': {'fpr': fpr_gru[::max(1, len(fpr_gru)//100)].tolist(), 'tpr': tpr_gru[::max(1, len(tpr_gru)//100)].tolist()},
                    'std_tf': {'fpr': fpr_std[::max(1, len(fpr_std)//100)].tolist(), 'tpr': tpr_std[::max(1, len(tpr_std)//100)].tolist()}
                },
                'pr_curves': {
                    'causal_tf': {'recall': rec_c[::max(1, len(rec_c)//100)].tolist(), 'precision': prec_c[::max(1, len(prec_c)//100)].tolist()}
                }
            }
            
    # Calculate Mean +/- Std for all models
    summary_table = {}
    for model_name, metric_list in all_results.items():
        summary_table[model_name] = {
            'auroc_mean': round(float(np.mean([m['auroc'] for m in metric_list])), 4),
            'auroc_std': round(float(np.std([m['auroc'] for m in metric_list], ddof=1)), 4),
            'auprc_mean': round(float(np.mean([m['auprc'] for m in metric_list])), 4),
            'auprc_std': round(float(np.std([m['auprc'] for m in metric_list], ddof=1)), 4),
            'f1_mean': round(float(np.mean([m['f1'] for m in metric_list])), 4),
            'f1_std': round(float(np.std([m['f1'] for m in metric_list], ddof=1)), 4),
            'brier_mean': round(float(np.mean([m['brier'] for m in metric_list])), 4),
            'brier_std': round(float(np.std([m['brier'] for m in metric_list], ddof=1)), 4)
        }
        
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK SUMMARY (MEAN +/- STD OVER 5 SEEDS)")
    print("=" * 60)
    for model_name, s in summary_table.items():
        print(f"{model_name:35s} | AUROC: {s['auroc_mean']:.4f} +/- {s['auroc_std']:.4f} | AUPRC: {s['auprc_mean']:.4f} +/- {s['auprc_std']:.4f} | F1: {s['f1_mean']:.4f} +/- {s['f1_std']:.4f}")
        
    final_payload = {
        'summary': summary_table,
        'raw_runs': all_results,
        'seed42_details': seed42_payload
    }
    
    os.makedirs('g:/tazz/checkpoints', exist_ok=True)
    with open('g:/tazz/checkpoints/real_benchmark_results.json', 'w') as f:
        json.dump(final_payload, f, indent=2)
    print("\nSaved genuine experimental benchmark results to checkpoints/real_benchmark_results.json")

if __name__ == '__main__':
    run_all_benchmarks()
