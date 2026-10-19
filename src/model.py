import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalCausalTransformer(nn.Module):
    """
    Temporal Causal-Masked Transformer for Longitudinal EHR Readmission Prediction.
    Uses lower-triangular autoregressive causal masking to preserve chronological history
    and prevent future-to-past information leakage.
    """
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=2, seq_len=10, dropout=0.1):
        super(TemporalCausalTransformer, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Feature Projection
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        # Transformer Causal Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 30-Day Readmission Risk Classifier Head
        self.readmission_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        # Auxiliary Biomarker Surrogate Head (exploratory)
        self.surrogate_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x, return_latent=False):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _ = x.size()
        h = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        mask = self._generate_square_subsequent_mask(seq_len).to(x.device)
        
        latent = self.transformer_encoder(h, mask=mask)
        pooled = latent[:, -1, :] # final encounter representation
        
        logits = self.readmission_classifier(pooled)
        surrogate_pred = self.surrogate_head(latent).squeeze(-1)
        
        res = {'logits': logits, 'surrogate': surrogate_pred}
        if return_latent:
            res['latent'] = latent
            res['pooled'] = pooled
        return res

class StandardTransformer(nn.Module):
    """
    Standard Non-Causal Transformer Encoder (Ablation / Baseline: no causal masking).
    """
    def __init__(self, input_dim=9, d_model=64, nhead=4, num_layers=2, seq_len=10, dropout=0.1):
        super(StandardTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.readmission_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h = self.embedding(x) + self.pos_encoder[:, :seq_len, :]
        latent = self.transformer_encoder(h) # No subsequent mask
        pooled = latent[:, -1, :]
        logits = self.readmission_classifier(pooled)
        return {'logits': logits}

class GRUBaseline(nn.Module):
    """
    Recurrent Gated Recurrent Unit (GRU) Baseline for longitudinal sequences.
    """
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=2, dropout=0.1):
        super(GRUBaseline, self).__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)
        )

    def forward(self, x):
        out, h_n = self.gru(x)
        last_step = out[:, -1, :]
        logits = self.classifier(last_step)
        return {'logits': logits}
