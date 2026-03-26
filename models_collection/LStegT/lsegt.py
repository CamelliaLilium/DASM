import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ..qim_transformer import HessianCompatibleTransformerLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout_p=0.5):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout_p)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        # Add positional encoding, pe will be broadcasted.
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Model1(nn.Module):
    def __init__(self, args):
        super(Model1, self).__init__()
        self.embedding = nn.Embedding(256, args.d_model)
        # CRITICAL FIX: Lower dropout throughout the model to reduce train/eval discrepancy
        # Original dropout=0.5 causes 2x scale difference between train and eval
        effective_dropout = min(args.dropout, 0.1)
        self.position_embedding = PositionalEncoding(args.d_model, max_len=args.max_len * 10, dropout_p=effective_dropout)
        self.transformer_layers = nn.ModuleList([
            HessianCompatibleTransformerLayer(args.d_model, args.num_heads, args.d_ff, effective_dropout)
            for _ in range(args.num_layers)
        ])
        self.pooling = nn.AdaptiveAvgPool1d(1)
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights to prevent vanishing gradients"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Embedding):
                # CRITICAL: Larger std for embedding to increase feature diversity (same fix as KFEF)
                nn.init.normal_(m.weight, mean=0.0, std=1.0)

    def forward(self, x):
        x = x.long()
        # Input x shape: (batch_size, 100, 7)
        emb_x = self.embedding(x)  # Shape: (batch_size, 100, 7, d_model)

        # Reshape to (batch_size, 700, d_model)
        emb_x = emb_x.view(emb_x.size(0), -1, emb_x.size(3))

        # Permute to (700, batch_size, d_model) for LStegT's Positional Encoding
        emb_x_permuted = emb_x.permute(1, 0, 2)

        # Add positional encoding
        emb_x_pos = self.position_embedding(emb_x_permuted)

        # Permute back to (batch_size, 700, d_model) for Hessian-compatible Transformer
        emb_x = emb_x_pos.permute(1, 0, 2)

        # Pass through Hessian-compatible transformer layers
        for layer in self.transformer_layers:
            emb_x = layer(emb_x)

        # Pooling expects (batch, dim, seq_len)
        outputs = self.pooling(emb_x.permute(0, 2, 1)).squeeze(2)
        return outputs


class Classifier1(nn.Module):
    """分类器 - 支持返回特征用于对比学习（CSAM）"""
    def __init__(self, args):
        super(Classifier1, self).__init__()
        self.args = args
        self.model1 = Model1(args)
        # CRITICAL FIX: Lower dropout from 0.5 to 0.1 to reduce train/eval discrepancy
        # High dropout causes huge behavior difference between train (50% neurons off) and eval (all on)
        self.dropout = nn.Dropout(min(args.dropout, 0.1))
        # CRITICAL FIX: Use LayerNorm instead of BatchNorm
        # BatchNorm's running statistics are unstable with small datasets (4000 samples)
        # LayerNorm has identical behavior in train and eval mode
        self.feature_norm = nn.LayerNorm(args.d_model)
        self.fc = nn.Linear(args.d_model, args.num_class)
        # Initialize classifier weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # CRITICAL FIX: Initialize bias with small random values instead of zero
                    # Zero bias causes systematic bias towards one class (especially in validation)
                    # Small random bias helps break symmetry and prevents all predictions being the same class
                    nn.init.normal_(m.bias, mean=0.0, std=0.01)

    def forward(self, x, return_features=False):
        features = self.model1(x)  # (batch, d_model)
        
        # # ========== DEBUG: Check feature extraction output ==========
        # if hasattr(self, '_debug_enabled') and self._debug_enabled:
        #     # CRITICAL: Check variance ACROSS BATCH (not entire tensor)
        #     features_batch_std = features.std(dim=0).mean().item()
        #     batch_size = features.shape[0]
        #     print(f"DEBUG: LStegT features batch_std={features_batch_std:.6f} (batch={batch_size})")
        #     # Only warn if batch is large enough for meaningful statistics
        #     if batch_size >= 32 and features_batch_std < 0.01:
        #         print(f"CRITICAL: Features nearly identical! batch_std={features_batch_std:.6f}")
        # # ========== END DEBUG ==========
        
        # CRITICAL FIX: Normalize features before classification to stabilize training
        # LayerNorm (not BatchNorm) ensures identical behavior in train and eval mode
        features = self.feature_norm(features)
        
        x = self.dropout(features)
        logits = self.fc(x)
        # Return logits for CrossEntropyLoss compatibility (used by CSAM)
        # For backward compatibility, return softmax when return_features=False and not using CSAM
        if return_features:
            return logits, features
        # Check if we're in a context that expects logits (CSAM training)
        # For now, return logits to be compatible with CSAM and standard training
        return logits


