import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class RealUCIDiabetesDataset(Dataset):
    """
    Robust EHR Dataset loader for UCI Diabetes 130-US Hospitals (diabetic_data.csv).
    
    Leakage prevention measures:
    1. Filter out hospice and in-hospital deaths (discharge_disposition_id in [11, 13, 14, 19, 20, 21]).
    2. Strict Patient-Level split: patient trajectories are strictly isolated to train/val/test splits.
    3. Proper 30-day readmission binary target: 1 if readmitted == '<30', else 0 ('>30' and 'NO').
    4. Chronological sequence per patient strictly bounded up to step t.
    """
    def __init__(self, patient_ids, df_cleaned, seq_len=10):
        self.seq_len = seq_len
        self.patient_ids = patient_ids
        
        feature_cols = [
            'age_num', 'los_norm', 'lab_norm', 'proc_norm', 'meds_norm',
            'outpatient_norm', 'emergency_norm', 'inpatient_norm', 'regimen_norm'
        ]
        
        # Filter dataframe to only include the designated patient IDs
        df_subset = df_cleaned[df_cleaned['patient_nbr'].isin(patient_ids)].copy()
        grouped = df_subset.groupby('patient_nbr')
        
        self.samples_features = []
        self.samples_hba1c = []
        self.samples_readm = []
        self.valid_patient_nbrs = []
        
        for p_id, group in grouped:
            feat_matrix = group[feature_cols].values
            hba1c_vec = group['hba1c_val'].values
            readm_label = group['readm_30d'].values[-1] # Status on final longitudinal encounter
            
            n_obs = len(group)
            if n_obs >= seq_len:
                seq_f = feat_matrix[:seq_len]
                seq_h = hba1c_vec[:seq_len]
            else:
                pad_len = seq_len - n_obs
                last_f = feat_matrix[-1:]
                last_h = hba1c_vec[-1:]
                seq_f = np.vstack([feat_matrix] + [last_f] * pad_len)
                seq_h = np.concatenate([hba1c_vec] + [last_h] * pad_len)
                
            self.samples_features.append(seq_f)
            self.samples_hba1c.append(seq_h)
            self.samples_readm.append(readm_label)
            self.valid_patient_nbrs.append(p_id)
            
        self.samples_features = np.array(self.samples_features, dtype=np.float32)
        self.samples_hba1c = np.array(self.samples_hba1c, dtype=np.float32)
        self.samples_readm = np.array(self.samples_readm, dtype=np.int64)

    def __len__(self):
        return len(self.samples_readm)

    def __getitem__(self, idx):
        return {
            'features': torch.tensor(self.samples_features[idx], dtype=torch.float32),
            'hba1c_trajectory': torch.tensor(self.samples_hba1c[idx], dtype=torch.float32),
            'readmission_label': torch.tensor(self.samples_readm[idx], dtype=torch.long)
        }

def load_and_preprocess_uci_data(csv_path='g:/tazz/data_uci/diabetic_data.csv'):
    """
    Load raw CSV, apply clinical filters, and preprocess tabular attributes.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    # 1. Exclude hospice discharges and in-hospital mortality
    # IDs: 11 (Expired), 13 (Hospice/home), 14 (Hospice/medical facility), 
    # 19 (Expired at home), 20 (Expired in medical facility), 21 (Expired unknown place)
    hospice_death_ids = [11, 13, 14, 19, 20, 21]
    df = df[~df['discharge_disposition_id'].isin(hospice_death_ids)].copy()
    
    # 2. Map Age brackets '[50-60)' -> midpoint / 100
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    df['age_num'] = df['age'].map(age_map).fillna(60) / 100.0
    
    # 3. Scale Numerical Features (using clinical max upper bounds)
    df['los_norm'] = df['time_in_hospital'] / 14.0
    df['lab_norm'] = df['num_lab_procedures'] / 132.0
    df['proc_norm'] = df['num_procedures'] / 6.0
    df['meds_norm'] = df['num_medications'] / 81.0
    df['outpatient_norm'] = df['number_outpatient'] / 42.0
    df['emergency_norm'] = df['number_emergency'] / 76.0
    df['inpatient_norm'] = df['number_inpatient'] / 21.0
    
    # 4. Medication regimen encoding (0: Standard Care, 0.5: Oral, 1.0: Insulin)
    def get_regimen(row):
        if row['insulin'] in ['Up', 'Down', 'Steady']:
            return 1.0
        elif row['metformin'] in ['Up', 'Down', 'Steady'] or row['diabetesMed'] == 'Yes':
            return 0.5
        return 0.0
        
    df['regimen_norm'] = df.apply(get_regimen, axis=1)
    
    # 5. HbA1c ordinal surrogate mapping (for exploratory multi-task branch)
    a1c_map = {'Norm': 6.0, '>7': 7.5, '>8': 9.0}
    df['hba1c_val'] = df['A1Cresult'].map(a1c_map).fillna(7.0)
    
    # 6. Readmission Target: STRICT 30-Day Readmission (<30 only)
    df['readm_30d'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
    
    return df

def get_train_val_test_datasets(csv_path='g:/tazz/data_uci/diabetic_data.csv', seed=42, seq_len=10):
    """
    Split data strictly by unique patient_nbr (70% train, 15% val, 15% test).
    """
    df_cleaned = load_and_preprocess_uci_data(csv_path)
    unique_patients = df_cleaned['patient_nbr'].unique()
    
    # Step 1: Patient-level train (70%) vs temp (30%)
    train_pts, temp_pts = train_test_split(unique_patients, test_size=0.30, random_state=seed)
    
    # Step 2: Split temp into val (15%) and test (15%)
    val_pts, test_pts = train_test_split(temp_pts, test_size=0.50, random_state=seed)
    
    train_ds = RealUCIDiabetesDataset(train_pts, df_cleaned, seq_len=seq_len)
    val_ds = RealUCIDiabetesDataset(val_pts, df_cleaned, seq_len=seq_len)
    test_ds = RealUCIDiabetesDataset(test_pts, df_cleaned, seq_len=seq_len)
    
    return train_ds, val_ds, test_ds
