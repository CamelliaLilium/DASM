import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.5)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Get input sequence length
        seq_len = x.size(1)

        # Truncate positional encoding to match input sequence length
        pe = self.pe[:, :seq_len, :]

        # Add positional encoding to input tensor
        x = x + pe

        return x

class HAM(nn.Module):
    """Hybrid Attention Mechanism - from DVSF baseline implementation"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(HAM, self).__init__()
        
        # Multi head self attention mechanism
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout) 
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Convolutional layer for local feature extraction
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv_combined = nn.Conv1d(2 * d_model, d_model, kernel_size=3, padding=1)
        
        # Add regularization term for weight decay
        self.weight_decay = 1e-5

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Convolutional processing for local features
        src = src.transpose(0, 1).transpose(1, 2)  # (S, B, E) -> (B, E, S)
        src2 = self.conv1(src)
        src2 = F.gelu(src2) 
        src2 = self.conv2(src2)
        src_cnn = src + self.dropout2(src2)  
        src_cnn = src_cnn.transpose(1, 2).transpose(0, 1)  # (B, E, S) -> (S, B, E)
        src = src.transpose(1, 2).transpose(0, 1)  # (B, E, S) -> (S, B, E)

        # Feed forward network
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        
        # Combine CNN and FFN features
        concatenated = torch.cat((src_cnn, self.dropout2(src2)), dim=2)
        concatenated = concatenated.transpose(0, 1).transpose(1, 2)

        output = self.conv_combined(concatenated)
        output = output.transpose(1, 2).transpose(0, 1)

        src = src + self.dropout2(output)
        src = self.norm2(src)  
        
        return src

def apply_regularization(model, weight_decay):
    """Apply weight decay regularization directly to parameters (DVSF style)"""
    for param in model.parameters():
        param.data = param.data - weight_decay * param.data

class HAM_multi(nn.Module):
    """Multi-layer HAM encoder"""
    def __init__(self, encoder_layer, num_layers):
        super(HAM_multi, self).__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            # Apply regularization as in baseline
            apply_regularization(self.layers[i], self.layers[i].weight_decay)

        return output

class Model_DVSF(nn.Module):
    """DVSF Model - feature extraction backbone"""
    def __init__(self, args):
        super(Model_DVSF, self).__init__()
        self.args = args
        
        self.embedding = nn.Embedding(256, args.d_model)
        self.position_embedding = PositionalEncoding(args.d_model, args.max_len)
        
        # Use HAM instead of standard transformer layers
        self.ham_encoder_layer = HAM(d_model=args.d_model, nhead=args.num_heads, 
                                    dim_feedforward=args.d_ff, dropout=args.dropout)
        self.ham_encoder = HAM_multi(self.ham_encoder_layer, num_layers=args.num_layers)
        
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.long()
        emb_x = self.embedding(x)
        
        # Add positional encoding
        emb_x += self.position_embedding(emb_x)

        # Reshape dimensions: (batch_size, 100, 3or7, d_model) -> (batch_size, 300or700, d_model)
        emb_x = emb_x.view(emb_x.size(0), -1, emb_x.size(3))
        
        # Permute for HAM: (batch_size, seq, d_model) -> (seq, batch_size, d_model)
        emb_x = emb_x.permute(1, 0, 2)
        
        # Pass through HAM encoder
        outputs = self.ham_encoder(emb_x)
        
        # Pooling: (seq, batch_size, d_model) -> (batch_size, d_model)
        outputs = self.pooling(outputs.permute(1, 2, 0)).squeeze(2)
        
        return outputs

class Classifier1(nn.Module):
    """DVSF Classifier - standard interface (for non-contrastive training)"""
    def __init__(self, args):
        super(Classifier1, self).__init__()
        self.args = args
        self.model = Model_DVSF(args)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.d_model, args.num_class)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)

class DVSF_Classifier_CL(nn.Module):
    """DVSF Classifier with Contrastive Learning (for DVSF runner)
    
    This version implements the triplet forward pass as in baseline:
    Input: batch of samples arranged as [cover_0, cover_1, steg_0, cover_2, cover_3, steg_1, ...]
    Output: (features_unsup, logits_cover, logits_steg, features_supervised)
    """
    def __init__(self, args):
        super(DVSF_Classifier_CL, self).__init__()
        self.args = args
        self.model = Model_DVSF(args)
        self.dropout = nn.Dropout(args.dropout)
        self.fc = nn.Linear(args.d_model, args.num_class)

    def forward(self, x):
        # Extract features for all samples in batch
        x_unsup = self.model(x)
        
        # Split features according to triplet arrangement
        # Every 3 samples: [cover_i, cover_i+1, steg_i]
        # Extract features at positions 0, 2, 4, ... (cover samples)
        # Extract features at positions 2, 5, 8, ... (steg samples)
        batch_size_triplet = x_unsup.size(0) // 3
        
        x_sup_1 = torch.zeros(batch_size_triplet, x_unsup.size(1)).to(x_unsup.device)
        x_sup_2 = torch.zeros(batch_size_triplet, x_unsup.size(1)).to(x_unsup.device)
        
        for i in range(batch_size_triplet):
            x_sup_1[i] = x_unsup[3 * i]      # Cover features
            x_sup_2[i] = x_unsup[3 * i + 2]  # Steg features
        
        # Store features for contrastive loss
        x_feats = x_sup_1
        
        # Classification heads
        x_sup_1 = self.dropout(x_sup_1)
        x_sup_1 = self.fc(x_sup_1)
        x_sup_1 = F.softmax(x_sup_1, dim=1)
        
        x_sup_2 = self.dropout(x_sup_2)
        x_sup_2 = self.fc(x_sup_2)
        x_sup_2 = F.softmax(x_sup_2, dim=1)

        return x_unsup, x_sup_1, x_sup_2, x_feats
