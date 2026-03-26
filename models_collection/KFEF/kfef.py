import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

# Baseline KFEF Multi-Head Attention with Random Masking (aligned with baseline)
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, num_heads * d_k)
        self.W_K = nn.Linear(d_model, num_heads * d_k)
        self.W_V = nn.Linear(d_model, num_heads * d_v)
        self.W_O = nn.Linear(num_heads * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn, V)
        return output, attn

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_Q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)

        # Scaled dot-product attention
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.scaled_dot_product_attention(Q, K, V, mask=mask)

        # Concatenation of the heads
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)

        # Final linear layer
        x = self.W_O(x)
        return x


# Baseline KFEF Positional Encoding (aligned with baseline)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        # Simplified positional encoding addition (aligned with baseline)
        x = x + self.pe[:seq_len, :].transpose(0, 1)  # (seq_len, d_model) -> (1, seq_len, d_model)
        return self.dropout(x)

# Baseline KFEF Model with Random Masking (aligned with baseline)
class BaselineKFEFModel(nn.Module):
    def __init__(self, args, input_dim):
        super(BaselineKFEFModel, self).__init__()
        self.args = args
        self.input_dim = input_dim
        
        # 关键修复：离散 codebook 索引必须用 Embedding！
        # 输入是 (batch, seq, input_dim)，每个值是 0-255 的离散索引
        # 需要先展平成 (batch, seq*input_dim)，embedding 后再 reshape
        self.embedding = nn.Embedding(256, args.d_model)
        
        # max_len 需要足够大以容纳 seq_len * input_dim (例如 100*7=700)
        max_seq_len = getattr(args, 'max_len', 100) * input_dim
        self.position_embedding = PositionalEncoding(args.d_model, dropout=0.5, max_len=max_seq_len)
        self.attentions = nn.ModuleList([
            MultiHeadAttention(8, args.d_model, 16, 8) for _ in range(args.num_layers)
        ])
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # CRITICAL FIX: Disable mask for end-to-end training
        # Mask was designed for pre-training frozen feature extractors in baseline
        # For end-to-end training from scratch, mask causes feature collapse
        # because eval mode has no mask, making all samples produce identical outputs
        self.mask_prob = getattr(args, 'mask_prob', 0.0)  # Disabled

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, input_dim) 例如 (64, 100, 3) 或 (64, 100, 4)
        # 每个值是 0-255 的离散 codebook 索引
        
        batch_size, seq_len, feat_dim = x.shape
        
        # 关键修复：正确处理多维离散输入
        # 先 reshape 成 (batch, seq*feat) 再 embedding
        x_flat = x.long().reshape(batch_size, seq_len * feat_dim)  # (batch, seq*feat)
        emb_x = self.embedding(x_flat)  # (batch, seq*feat, d_model)
        
        # Add positional encoding
        emb_x = self.position_embedding(emb_x)
        
        
        # Attention causes feature collapse when training from scratch
        # Use simple feed-forward instead
        if self.mask_prob > 0 and hasattr(self, 'attentions') and len(self.attentions) > 0:
            # Only use attention if explicitly enabled
            for attention in self.attentions:
                residual = emb_x
                attn_output = attention(emb_x, emb_x, emb_x, mask=mask)
                emb_x = residual + attn_output
        
        
        # Global average pooling
        outputs = self.pooling(emb_x.permute(0, 2, 1)).squeeze(2)
        return outputs

# Baseline KFEF Classifier (aligned with baseline dual-branch architecture)
class BaselineKFEFClassifier(nn.Module):
    def __init__(self, args):
        super(BaselineKFEFClassifier, self).__init__()
        self.args = args
        self.training_stage = getattr(args, 'training_stage', 'end_to_end')  # 'stage1', 'stage2', 'end_to_end'
        
        # QIM branch: processes first 3 dimensions (baseline architecture)
        self.qim_model = BaselineKFEFModel(args, input_dim=3)
        
        # PMS branch: processes last 4 dimensions (baseline architecture)
        self.pms_model = BaselineKFEFModel(args, input_dim=4)
        
        # Classification heads for each branch (baseline approach)
        self.qim_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(args.d_model, args.num_class)
        )
        
        self.pms_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(args.d_model, args.num_class)
        )
        
        # BatchNorm to enhance feature contrast before classification
        self.feature_norm = nn.BatchNorm1d(args.d_model * 2)
        
        # MLP classifier with proper hidden layer
        self.fusion_classifier = nn.Sequential(
            nn.Linear(args.d_model * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, args.num_class)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Embedding):
                # CRITICAL: Larger std for embedding to increase feature diversity
                nn.init.normal_(m.weight, mean=0.0, std=1.0)

    def forward(self, x):
        # x shape: (batch_size, seq_len, 7)
        
        # Split input according to baseline: QIM (3-dim) + PMS (4-dim)
        x_qim = x[:, :, :3]   # First 3 dimensions for QIM branch
        x_pms = x[:, :, 3:]   # Last 4 dimensions for PMS branch (3:7 = 4 dims)

        # Extract features from each branch with random masking
        qim_features = self.qim_model(x_qim)
        pms_features = self.pms_model(x_pms)
        
        # # ========== DEBUG: Check feature extraction output ==========
        # if hasattr(self, '_debug_enabled') and self._debug_enabled:
        #     # CRITICAL: Check variance ACROSS BATCH (not entire tensor)
        #     qim_batch_std = qim_features.std(dim=0).mean().item()
        #     pms_batch_std = pms_features.std(dim=0).mean().item()
        #     batch_size = qim_features.shape[0]
        #     print(f"DEBUG: qim batch_std={qim_batch_std:.6f}, pms batch_std={pms_batch_std:.6f} (batch={batch_size})")
        #     # Only warn if batch is large enough for meaningful statistics
        #     if batch_size >= 32 and (qim_batch_std < 0.01 or pms_batch_std < 0.01):
        #         print(f"CRITICAL: Features nearly identical! qim={qim_batch_std:.6f}, pms={pms_batch_std:.6f}")
        # # ========== END DEBUG ==========

        if self.training_stage == 'stage1':
            # Stage 1: Train only QIM branch
            return self.qim_classifier(qim_features)
        elif self.training_stage == 'stage2':
            # Stage 2: Train only PMS branch
            return self.pms_classifier(pms_features)
        else:
            # End-to-end or fusion training
            # Concatenate features from both branches
            fused_features = torch.cat((qim_features, pms_features), dim=1)
            # BatchNorm to enhance feature contrast
            fused_features = self.feature_norm(fused_features)
            # Final classification
            output = self.fusion_classifier(fused_features)
            return output

    def set_training_stage(self, stage):
        """Set training stage: 'stage1', 'stage2', or 'end_to_end'"""
        self.training_stage = stage
        
    def freeze_branch_models(self):
        """Freeze QIM and PMS branch models for fusion training"""
        for param in self.qim_model.parameters():
            param.requires_grad = False
        for param in self.pms_model.parameters():
            param.requires_grad = False
        for param in self.qim_classifier.parameters():
            param.requires_grad = False
        for param in self.pms_classifier.parameters():
            param.requires_grad = False
            
    def unfreeze_all(self):
        """Unfreeze all parameters for end-to-end training"""
        for param in self.parameters():
            param.requires_grad = True

# Alias for backward compatibility
KFEFClassifier = BaselineKFEFClassifier 