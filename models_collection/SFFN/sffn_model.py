#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
Model definitions for SFFN.
This version is simplified for standard, end-to-end training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLoss(nn.Module):
    """
    Original SFFN loss function that expects Softmax outputs and uses binary cross entropy.
    """
    def __init__(self, n_class):
        super(SimpleLoss, self).__init__()
        self.n_class = n_class

    def forward(self, pred, label):
        # one-hot encoding
        label_onehot = torch.FloatTensor(label.size()[0], self.n_class)
        if torch.cuda.is_available():
            label = label.cuda()
            label_onehot = label_onehot.cuda()
        label_onehot.zero_()
        label_onehot.scatter_(1, torch.unsqueeze(label, 1), 1)
        loss_main = F.binary_cross_entropy(pred, label_onehot)
        return loss_main


class CLS(nn.Module):
    """
    A simple classifier module.
    Matches the original SFFN implementation with Softmax.
    """
    def __init__(self, in_size, n_class):
        super(CLS, self).__init__()
        # Restore original SFFN architecture with Softmax
        self.classifier = nn.Sequential(nn.Linear(in_size, n_class), nn.Softmax(dim=1))

    def forward(self, x):
        out = self.classifier(x)
        return out


class CNN(nn.Module):
    """
    The CNN module for feature fusion.
    """
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
        x = torch.squeeze(x, dim=-1)
        x = torch.squeeze(x, dim=-1)
        return x


class SFFN_Model(nn.Module):
    """
    The main SFFN model, faithful reproduction of original SFFN architecture.
    It combines two LSTMs and a CNN with three classifiers.
    """
    def __init__(self, opt):
        super(SFFN_Model, self).__init__()
        # Store necessary parameters from opt
        self.input_dim1 = 3
        self.input_dim2 = 4
        self.num_layers = opt.num_layers
        self.hidden_size = opt.hidden_num # Use hidden_num from main args
        self.n_class = opt.num_class # Use num_class from main args
        
        self.rnn1 = nn.LSTM(input_size=self.input_dim1, num_layers=self.num_layers, hidden_size=self.hidden_size,
                            batch_first=True)
        self.rnn2 = nn.LSTM(input_size=self.input_dim2, num_layers=self.num_layers, hidden_size=self.hidden_size,
                            batch_first=True)
        self.cnn = CNN()
        self.cls = CLS(in_size=128, n_class=self.n_class)
        self.cls1 = CLS(in_size=self.hidden_size, n_class=self.n_class)  # Added: auxiliary classifier for rnn1
        self.cls2 = CLS(in_size=self.hidden_size, n_class=self.n_class)  # Added: auxiliary classifier for rnn2

    def forward(self, x, domain_id=None):
        # x shape: (batch_size, seq_len, 7)
        # Slice the input into two feature streams - exactly like original SFFN
        x1 = x[:, :, 0:self.input_dim1]
        x2 = x[:, :, self.input_dim1:self.input_dim1 + self.input_dim2]

        # Process through respective LSTMs
        embed1, _ = self.rnn1(x1)  # (batch, seq_len, hidden_size)
        embed2, _ = self.rnn2(x2)  # (batch, seq_len, hidden_size)

        # Extract last timestep for auxiliary classifiers - exactly like original SFFN
        embed1_t = embed1[:, -1, :]  # (batch, hidden_size)
        embed2_t = embed2[:, -1, :]  # (batch, hidden_size)

        # --- Gradient Gating for SASM ---
        # When training with SASM, we prevent gradient cross-contamination
        # by detaching the output of the non-expert RNN from the computation graph.
        if domain_id is not None:
            if domain_id == 0:  # QIM domain: only train rnn1
                embed2 = embed2.detach()
                embed2_t = embed2_t.detach()
            elif domain_id == 1:  # PMS domain: only train rnn2
                embed1 = embed1.detach()
                embed1_t = embed1_t.detach()
            # For other domains (e.g., LSB, domain_id=2) or if domain_id is None (non-SASM),
            # gradients flow to both RNNs. This ensures non-SASM training is unaffected.

        # CNN fusion using full sequences - exactly like original SFFN
        embed_cat = torch.stack((embed1, embed2), 1)  # (batch, 2, seq_len, hidden_size)
        embed = self.cnn(embed_cat)  # (batch, 128)

        # Final classification - handle potential 1D case
        if embed.dim() == 1:
            embed = embed.unsqueeze(0)
            
        # Three predictions exactly like original SFFN
        pred = self.cls(embed)      # Main prediction from CNN fusion
        pred1 = self.cls1(embed1_t) # Auxiliary prediction from rnn1 last timestep
        pred2 = self.cls2(embed2_t) # Auxiliary prediction from rnn2 last timestep
        
        return pred, pred1, pred2 