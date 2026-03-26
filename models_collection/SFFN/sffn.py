import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimpleLoss(nn.Module):
    """Baseline SFFN loss function (binary cross entropy with one-hot)"""
    def __init__(self, n_class):
        super(SimpleLoss, self).__init__()
        self.n_class = n_class

    def forward(self, pred, label):
        # Convert to one-hot encoding
        label_onehot = torch.FloatTensor(label.size()[0], self.n_class)
        if torch.cuda.is_available():
            label = label.cuda()
            label_onehot = label_onehot.cuda()
        label_onehot.zero_()
        label_onehot.scatter_(1, torch.unsqueeze(label, 1), 1)
        loss_main = F.binary_cross_entropy(pred, label_onehot)
        return loss_main

class CLS(nn.Module):
    """Baseline SFFN classifier head"""
    def __init__(self, in_size, n_class):
        super(CLS, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_size, n_class), 
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        out = self.classifier(x)
        return out

class CNN(nn.Module):
    """Baseline SFFN CNN fusion module"""
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.mp = nn.MaxPool2d(kernel_size=2)
        self.ap = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mp(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.relu(x)
        x = self.ap(x)
        x = torch.squeeze(x)
        return x

class BaselineSFFNModel(nn.Module):
    """Baseline SFFN Model (aligned with original SFFN.py)"""
    def __init__(self, args):
        super(BaselineSFFNModel, self).__init__()
        self.args = args
        
        # Dual-branch LSTM (baseline architecture)
        self.rnn1 = nn.LSTM(
            input_size=3,  # QIM branch: 3 dimensions
            num_layers=2,  # Fixed to 2 layers as in baseline
            hidden_size=50,  # Fixed to 50 as in baseline
            batch_first=True
        )
        
        self.rnn2 = nn.LSTM(
            input_size=4,  # PMS branch: 4 dimensions  
            num_layers=2,  # Fixed to 2 layers as in baseline
            hidden_size=50,  # Fixed to 50 as in baseline
            batch_first=True
        )
        
        # CNN fusion module (baseline)
        self.cnn = CNN()
        
        # Classification heads (baseline)
        self.cls = CLS(in_size=128, n_class=args.num_class)  # Main classifier
        self.cls1 = CLS(in_size=50, n_class=args.num_class)  # Branch 1 classifier
        self.cls2 = CLS(in_size=50, n_class=args.num_class)  # Branch 2 classifier
        
        # Loss function (baseline)
        self.criterion = SimpleLoss(args.num_class)

    def load_state_dict_pretrain1(self, state_dict):
        """Load pretrained RNN1 and CLS1 (Stage 1)"""
        self.rnn1.load_state_dict(state_dict[0])
        self.cls1.load_state_dict(state_dict[1])

    def load_state_dict_pretrain2(self, state_dict):
        """Load pretrained RNN2 and CLS2 (Stage 2)"""
        self.rnn2.load_state_dict(state_dict[0])
        self.cls2.load_state_dict(state_dict[1])

    def load_state_dict(self, state_dict):
        """Load complete model state (baseline format)"""
        self.rnn1.load_state_dict(state_dict[0])
        self.cls1.load_state_dict(state_dict[1])
        self.rnn2.load_state_dict(state_dict[2])
        self.cls2.load_state_dict(state_dict[3])
        self.cnn.load_state_dict(state_dict[4])
        self.cls.load_state_dict(state_dict[5])

    def state_dict(self):
        """Return baseline-compatible state dict format"""
        state_dict = [
            self.rnn1.state_dict(), 
            self.cls1.state_dict(), 
            self.rnn2.state_dict(), 
            self.cls2.state_dict(),
            self.cnn.state_dict(), 
            self.cls.state_dict()
        ]
        return state_dict

    def forward_loss(self, pred, label):
        """Compute loss using baseline loss function"""
        loss = self.criterion(pred, label)
        return loss

    def forward(self, x, return_aux=False):
        """Forward pass - return fused logits by default (optionally with aux outputs)."""
        # Split input: QIM (3-dim) + PMS (4-dim)
        x1 = x[:, :, 0:3]  # First 3 dimensions for RNN1
        x2 = x[:, :, 3:7]  # Last 4 dimensions for RNN2
        
        # Process through dual LSTM branches
        embed1, _ = self.rnn1(x1)
        embed2, _ = self.rnn2(x2)
        
        # Take last timestep output
        embed1_t = embed1[:, -1, :]
        embed2_t = embed2[:, -1, :]
        
        # Stack embeddings for CNN fusion (baseline approach)
        embed_cat = torch.stack((embed1, embed2), 1)  # Shape: (batch, 2, seq_len, hidden)
        
        # CNN fusion
        embed = self.cnn(embed_cat)
        
        # Main prediction (fused)
        pred = self.cls(embed)
        
        # Auxiliary predictions (individual branches)
        pred1 = self.cls1(embed1_t)
        pred2 = self.cls2(embed2_t)
        
        # Convert softmax outputs to logits for CrossEntropyLoss
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        pred_logits = torch.log(pred + eps)
        pred1_logits = torch.log(pred1 + eps)
        pred2_logits = torch.log(pred2 + eps)
        
        if return_aux:
            return pred_logits, pred1_logits, pred2_logits
        return pred_logits

# Alias for compatibility
SFFNClassifier = BaselineSFFNModel
