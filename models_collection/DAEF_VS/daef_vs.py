"""
DAEF-VS model wrapper for domain generalization framework.
Wraps the baseline DAEF-VS implementation into a standard interface.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add baseline path to import DAEF-VS
baseline_path = os.environ.get('DASM_DAEF_VS_BASELINE_PATH', os.path.join(PROJECT_ROOT, 'baseline_models', 'DAEF-VS'))
if baseline_path not in sys.path:
    sys.path.insert(0, baseline_path)

# Import components from baseline DAEF-VS
try:
    # Try importing without the device dependency
    import importlib.util
    spec = importlib.util.spec_from_file_location("daef_vs_baseline", 
                                                   os.path.join(baseline_path, "DAEF-VS.py"))
    daef_module = importlib.util.module_from_spec(spec)
    
    # Temporarily set device to avoid errors
    import torch as _torch
    if not hasattr(daef_module, 'device'):
        daef_module.device = _torch.device('cuda' if _torch.cuda.is_available() else 'cpu')
    
    spec.loader.exec_module(daef_module)
    
    PositionalEncoding = daef_module.PositionalEncoding
    Model_CL = daef_module.Model_CL
except Exception as e:
    print(f"Warning: Could not import DAEF-VS baseline: {e}")
    # Define fallback classes if import fails
    
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=100):
            super(PositionalEncoding, self).__init__()
            position = torch.arange(max_len).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe
            return x

    class Model_CL(nn.Module):
        """DAEF-VS base model (fallback implementation)"""
        def __init__(self, num_layers=1, d_model=64, nhead=8, bn_dim=700):
            super(Model_CL, self).__init__()
            self.embedding = nn.Embedding(256, d_model)
            self.position_embedding = PositionalEncoding(d_model)
            self.transformer_encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
            self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layers, num_layers=num_layers)
            self.bn = nn.BatchNorm1d(num_features=bn_dim)
            self.pooling = nn.AdaptiveAvgPool1d(1)
            
        def forward(self, x):
            x = x.long()
            emb_x = self.embedding(x)
            emb_x += self.position_embedding(emb_x)
            emb_x = emb_x.view(emb_x.size(0), -1, emb_x.size(3))
            emb_x = emb_x.permute(1, 0, 2)
            outputs = self.transformer_encoder(emb_x)
            outputs = self.bn(outputs.permute(1, 0, 2))
            outputs = self.pooling(outputs.permute(0, 2, 1)).squeeze(2)
            return outputs


class Classifier1(nn.Module):
    """DAEF-VS Classifier - standard interface (for unified training)
    
    This is a simplified version that works with the standard training loop.
    Uses flexible BN dimension based on input sequence length.
    """
    def __init__(self, args):
        super(Classifier1, self).__init__()
        self.args = args
        
        # Use args parameters or defaults
        d_model = getattr(args, 'd_model', 64)
        num_layers = getattr(args, 'num_layers', 1)
        nhead = getattr(args, 'num_heads', 8)
        
        # Calculate BN dimension: seq_len * d_model
        # For 100-frame sequences with 7 features -> 700 dims
        seq_len = getattr(args, 'max_len', 100)
        bn_dim = seq_len * d_model
        
        self.model = _init_model_cl(num_layers=num_layers, d_model=d_model, nhead=nhead, bn_dim=bn_dim)
        self.dropout = nn.Dropout(getattr(args, 'dropout', 0.5))
        self.fc = nn.Linear(d_model, args.num_class)

    def forward(self, x):
        """Standard forward pass for classification
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)
            
        Returns:
            Softmax probabilities of shape (batch_size, num_class)
        """
        x = self.model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, dim=1)


class DAEF_VS_Classifier_CL(nn.Module):
    """DAEF-VS Classifier with Contrastive Learning
    
    This version implements triplet forward pass for contrastive learning.
    Only used if you need the original contrastive learning approach.
    """
    def __init__(self, args):
        super(DAEF_VS_Classifier_CL, self).__init__()
        self.args = args
        
        d_model = getattr(args, 'd_model', 64)
        num_layers = getattr(args, 'num_layers', 1)
        nhead = getattr(args, 'num_heads', 8)
        seq_len = getattr(args, 'max_len', 100)
        bn_dim = seq_len * d_model
        
        self.model = _init_model_cl(num_layers=num_layers, d_model=d_model, nhead=nhead, bn_dim=bn_dim)
        self.dropout = nn.Dropout(getattr(args, 'dropout', 0.5))
        self.fc = nn.Linear(d_model, args.num_class)

    def forward(self, x):
        """Triplet forward pass
        
        Input is arranged as [cover_0, cover_1, steg_0, cover_2, cover_3, steg_1, ...]
        Returns: (features_unsup, logits_cover, logits_steg, features_sup)
        """
        x_unsup = self.model(x)
        
        batch_size_triplet = x_unsup.size(0) // 3
        device = x_unsup.device
        
        x_sup_1 = torch.zeros(batch_size_triplet, x_unsup.size(1)).to(device)
        x_sup_2 = torch.zeros(batch_size_triplet, x_unsup.size(1)).to(device)
        
        for i in range(batch_size_triplet):
            x_sup_1[i] = x_unsup[3 * i]      # Cover features
            x_sup_2[i] = x_unsup[3 * i + 2]  # Steg features
        
        x_feats = x_sup_1
        
        # Classification heads
        x_sup_1 = self.dropout(x_sup_1)
        logits_cover = self.fc(x_sup_1)
        logits_cover = F.softmax(logits_cover, dim=1)
        
        x_sup_2 = self.dropout(x_sup_2)
        logits_steg = self.fc(x_sup_2)
        logits_steg = F.softmax(logits_steg, dim=1)
        
        return x_unsup, logits_cover, logits_steg, x_feats


def _init_model_cl(num_layers=1, d_model=64, nhead=8, bn_dim=700):
    """Initialize baseline Model_CL with compatible kwargs."""
    import inspect
    signature = inspect.signature(Model_CL)
    kwargs = {}
    if "num_layers" in signature.parameters:
        kwargs["num_layers"] = num_layers
    if "d_model" in signature.parameters:
        kwargs["d_model"] = d_model
    if "nhead" in signature.parameters:
        kwargs["nhead"] = nhead
    if "bn_dim" in signature.parameters:
        kwargs["bn_dim"] = bn_dim
    return Model_CL(**kwargs)
