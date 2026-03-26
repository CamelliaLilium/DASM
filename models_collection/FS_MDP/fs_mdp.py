import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

dropout = 0.5


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, d_k):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, d_model)
        self.a = 1 / math.sqrt(d_k)

    def forward(self, x):
        bs, _, l_x = x.size()
        x = x.transpose(1, 2)

        k = self.k_linear(x).view(bs, l_x, self.n_head, self.d_k)
        q = self.q_linear(x).view(bs, l_x, self.n_head, self.d_k)
        v = self.v_linear(x).view(bs, l_x, self.n_head, self.d_k)
        q = q.permute(2, 0, 1, 3).contiguous().view(bs * self.n_head, l_x, self.d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(bs * self.n_head, l_x, self.d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(bs * self.n_head, l_x, self.d_k)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.a

        attn = F.softmax(attn, dim=2)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v)
        out = out.view(self.n_head, bs, l_x, self.d_k).permute(1, 2, 0, 3).contiguous().view(bs, l_x, self.d_model)
        out = self.fc(out)
        out = self.dropout(out)
        return out.transpose(1, 2)


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, dim=1, bias=True):
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=k, groups=in_ch,
                                            padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Wrong dimension for Depthwise Separable Convolution!")
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.depthwise_conv.bias, 0.0)
        nn.init.kaiming_normal_(self.depthwise_conv.weight)
        nn.init.constant_(self.pointwise_conv.bias, 0.0)

    def forward(self, x):
        return self.pointwise_conv(self.depthwise_conv(x))


class MA(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_k = d_model // n_head
        self.n_head = n_head
        self.conv1 = nn.Conv1d(in_channels=self.d_k, out_channels=600, kernel_size=3, padding=1)

    def forward(self, x):
        start = 0
        end = int(start + self.d_k)

        for i in range(self.n_head):
            x1 = x[:, start:end, :]
            out_conv1 = self.conv1(x1)
            ba = torch.bmm(x1, out_conv1).float()
            if i == 0:
                X = ba
            else:
                X = torch.cat((X, ba), 1)

            start += int(self.d_k)
            end += int(self.d_k)

        return X


class GtoL(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = MultiHeadAttention(600, 10, 600 // 10)
        self.conv1 = DepthwiseSeparableConv(600, 600, 3)

    def forward(self, x):
        X = x.transpose(1, 2)
        Xg = self.a1(X)

        Xl = self.conv1(Xg)

        return Xg, Xl, Xl


class LtoG(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = MultiHeadAttention(600, 10, 600 // 10)
        self.conv1 = DepthwiseSeparableConv(600, 600, 3)

    def forward(self, X):
        X = X.transpose(1, 2)
        Xl = self.conv1(X)

        Xg = self.a1(Xl)

        return Xg, Xl, Xg


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.block_1 = nn.LSTM(input_size=450, hidden_size=300, num_layers=2, batch_first=True, bidirectional=True)
        self.cq_att = MultiHeadAttention(300, 10, 300 // 10)
        self.a1 = MultiHeadAttention(300, 10, 300 // 10)
        self.a2 = MultiHeadAttention(300, 10, 300 // 10)
        self.f1 = nn.Sequential(
            nn.Linear(600 * 100, 100, bias=True),
            nn.BatchNorm1d(100)
        )
        self.f2 = nn.Sequential(
            nn.Linear(600 * 100, 100, bias=True),
            nn.BatchNorm1d(100)
        )

        channel = 200
        reduction = 2
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
        self.gtl1 = GtoL()
        self.ltg1 = LtoG()
        self.gtl2 = GtoL()
        self.ltg2 = LtoG()

        self.conv1 = DepthwiseSeparableConv(300, 300, 3)
        self.conv2 = DepthwiseSeparableConv(300, 300, 3)
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1)
        self.ca = ChannelAttention(3)
        self.classifier = nn.Sequential(
            nn.Linear(1200 * 100, 100, bias=True),
            nn.BatchNorm1d(100),
            nn.Sigmoid()
        )

    def forward(self, X_projected):
        batch, _, _ = X_projected.size()

        X, _ = self.block_1(X_projected.float())

        Xg1, Xl1, X1 = self.ltg1(X)
        Xg2, Xl2, X2 = self.gtl1(X)

        Xg = torch.cat((Xg1, Xg2), 2)
        Xl = torch.cat((Xl1, Xl2), 2)

        X = torch.cat((X1, X2), 2)
        Xg = Xg.unsqueeze(1)
        Xl = Xl.unsqueeze(1)
        X = X.unsqueeze(1)
        X = torch.cat((X, Xg, Xl), 1)
        ca = self.ca(X)
        X = ca * X
        X = self.conv3(X)
        X = X.squeeze(1)

        out_concat = X.contiguous().view(batch, -1)

        output = self.classifier(out_concat)
        
        # The original output is (batch, 100). The loss in main.py is BCELoss, which expects (N, 1) or (N,).
        # And accuracy is calculated with round, which suggests a binary problem.
        # This seems to be a multi-label classification problem instead of multi-class.
        # To adapt to the BCELoss in model_multi.py, we will take the mean.
        return output.mean(dim=1, keepdim=True)


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()

        self.block_1 = nn.LSTM(input_size=450, hidden_size=300, num_layers=3, batch_first=True, bidirectional=True)

        self.classifier = nn.Sequential(
            nn.Linear(1000*300*2, 1, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.projection_matrix1 = torch.nn.Parameter(torch.randn(128, 150))
        self.projection_matrix2 = torch.nn.Parameter(torch.randn(32, 150))
        self.projection_matrix3 = torch.nn.Parameter(torch.randn(32, 150))

    def forward(self, X):
        batch, seq, _ = X.size()
        l_1 = X[:, :, 0:128]
        l_2 = X[:, :, 128:160]
        l_3 = X[:, :, 160:192]
        l_1_embedding = l_1.matmul(self.projection_matrix1)
        l_2_embedding = l_2.matmul(self.projection_matrix2)
        l_3_embedding = l_3.matmul(self.projection_matrix3)
        X_projected = torch.cat((l_1_embedding, l_2_embedding, l_3_embedding), -1)
        return X_projected


class FS_MDP_Wrapper(nn.Module):
    def __init__(self, args):
        super(FS_MDP_Wrapper, self).__init__()
        self.embedding = Embedding()
        
        table_path = os.environ.get(
            'DASM_FS_MDP_TABLE_PATH',
            os.path.join(PROJECT_ROOT, 'models_collection', 'wordTable', 'table_best_chinese.pth')
        )
        
        if os.path.exists(table_path):
            print(f"Loading FS-MDP embedding weights from {table_path}")
            self.embedding.load_state_dict(torch.load(table_path, map_location='cpu'))
        else:
            print(f"Warning: FS-MDP embedding weight file not found at {table_path}. Using random initialization.")

        # Freeze embedding weights as in original code
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.classifier = Classifier()

    def forward(self, x):
        # x is expected to be one-hot encoded (N, seq, 192)
        x_projected = self.embedding(x)
        output = self.classifier(x_projected)
        return output 
