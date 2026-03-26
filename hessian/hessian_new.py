
#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import numpy as np

from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal


def get_params_grad_new(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        if param.grad is not None:
            grads.append(param.grad.clone().detach().requires_grad_(True))
        else:
            grads.append(torch.zeros_like(param, requires_grad=True))
        # grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads

def check_params_grads(model, params, grads):
    model_params = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == len(model_params), "参数数量不一致"
    assert len(grads) == len(model_params), "梯度数量不一致"
    for i, (p1, p2, g) in enumerate(zip(model_params, params, grads)):
        assert p1 is p2, f"第{i}个参数对象不一致"
        assert g.shape == p2.shape, f"第{i}个梯度shape不一致"
    print("params 和 grads 与模型参数完全一致")

class hessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, model, criterion, data, dataloader, cuda=True):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (data == None and
                                                         dataloader != None)

        self.model = model.eval()  # make model is in evaluation model
        self.criterion = criterion

        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True

        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            if self.device == 'cuda':
                self.inputs, self.targets = self.inputs.cuda(
                ), self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs = self.model(self.inputs)
            loss = self.criterion(outputs, self.targets)
            loss.backward(create_graph=True)
            # for name, param in model.named_parameters():
            #     print(name, param.requires_grad, param.grad is None)

        # this step is used to extract the parameters from the model
        params, gradsH = get_params_grad(self.model)
        self.params = params
        self.gradsH = gradsH
        # self.gradsH = [g if isinstance(g, torch.Tensor) else torch.zeros_like(p) for g, p in zip(gradsH, params)]  # gradient used for Hessian computation
        for param in self.model.parameters():
            param.grad = None

    def dataloader_hv_product(self, v):
        """
        计算 Hessian-向量乘积 Hv，使用 autograd.grad 避免内存泄漏。
        
        修复：使用 autograd.grad 替代 backward(create_graph=True)，
        避免 PyTorch 警告的引用循环内存泄漏问题。
        """
        assert isinstance(v, list) and all(isinstance(x, torch.Tensor) for x in v)

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(p.size()).to(device) for p in self.params]  # accumulate result
        
        # 内存清理计数器
        gc_interval = 10  # 每10个batch清理一次GPU缓存
        batch_count = 0
        
        for batch_data in self.data:
            # Handle both (inputs, labels) and (inputs, labels, algo_labels) formats
            inputs, labels = batch_data[0], batch_data[1]
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            outputs = self.model(inputs.to(device))
            
            # Handle different label formats safely
            if labels.dim() > 1 and labels.size(1) > 1:
                labels = labels[:, 1].long().to(device)
            else:
                labels = labels.long().to(device)
                
            loss = self.criterion(outputs, labels)
            
            # ============ 关键修复：使用 autograd.grad 替代 backward(create_graph=True) ============
            # PyTorch 官方推荐：避免 backward(create_graph=True) 导致的引用循环内存泄漏
            # 
            # 原理：Hv = ∇(∇L · v) = ∇(Σ gᵢvᵢ)
            # 第一步：计算梯度 g = ∇L（需要 create_graph=True 保留图）
            # 第二步：计算 Hv = ∇(g·v)（使用第一步的图，然后释放）
            
            # 第一步：计算梯度，保留计算图用于二阶导数
            # 注意：必须 retain_graph=True，因为需要用这个图来计算Hv
            gradsH = torch.autograd.grad(
                loss, 
                self.params, 
                create_graph=True,  # 保留图用于计算Hv
                retain_graph=True   # 必须保留！下一步还要用
            )
            
            # 第二步：计算 Hessian-向量乘积
            # 这里可以 retain_graph=False，因为不再需要这个图了
            Hv = torch.autograd.grad(
                gradsH,
                self.params,
                grad_outputs=v,
                only_inputs=True,
                retain_graph=False  # 计算完立即释放计算图
            )
            
            # 累积结果（detach避免保留计算图引用）
            THv = [
                THv1 + Hv1.detach() * float(tmp_num_data)
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)
            
            # ============ 显式清理，打破引用循环 ============
            # 清理模型梯度
            for param in self.model.parameters():
                param.grad = None
            
            # 删除中间变量，释放计算图引用
            del gradsH, Hv, outputs, loss
            
            # 定期清理GPU缓存（避免每次都清理影响性能）
            batch_count += 1
            if batch_count % gc_interval == 0:
                torch.cuda.empty_cache()

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue_tensor = group_product(THv, v)
        eigenvalue = eigenvalue_tensor.cpu().item() if isinstance(eigenvalue_tensor, torch.Tensor) else float(eigenvalue_tensor)
        return eigenvalue, THv

    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        
        内存优化：定期清理 GPU 缓存
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0
        gc_interval = 10  # 每10次迭代清理一次

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params
                 ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue_tensor = group_product(Hv, v)
                    tmp_eigenvalue = tmp_eigenvalue_tensor.cpu().item() if isinstance(tmp_eigenvalue_tensor, torch.Tensor) else float(tmp_eigenvalue_tensor)

                v = normalization(Hv)
                
                # 定期清理 GPU 缓存
                if (i + 1) % gc_interval == 0:
                    torch.cuda.empty_cache()

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1
            
            # 每计算完一个特征值，清理缓存
            torch.cuda.empty_cache()

        return eigenvalues, eigenvectors

    def trace(self, maxIter=100, tol=1e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        
        内存优化：定期清理 GPU 缓存
        """

        device = self.device
        trace_vhv = []
        trace = 0.
        gc_interval = 10  # 每10次迭代清理一次

        for i in range(maxIter):
            self.model.zero_grad()
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1

            if self.full_dataset:
                _, Hv = self.dataloader_hv_product(v)
            else:
                Hv = hessian_vector_product(self.gradsH, self.params, v)
            trace_tensor = group_product(Hv, v)
            trace_vhv.append(trace_tensor.cpu().item() if isinstance(trace_tensor, torch.Tensor) else float(trace_tensor))
            
            # 定期清理 GPU 缓存
            if (i + 1) % gc_interval == 0:
                torch.cuda.empty_cache()
            
            if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vhv
            else:
                trace = np.mean(trace_vhv)

        return trace_vhv

    def density(self, iter=100, n_v=1):
        """
        compute estimated eigenvalue density using stochastic lanczos algorithm (SLQ)
        iter: number of iterations used to compute trace
        n_v: number of SLQ runs
        
        内存优化：
        - 定期清理 GPU 缓存
        - 使用 detach() 避免计算图累积
        """

        device = self.device
        eigen_list_full = []
        weight_list_full = []

        for k in range(n_v):
            v = [
                torch.randint_like(p, high=2, device=device)
                for p in self.params
            ]
            # generate Rademacher random variables
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            # standard lanczos algorithm initlization
            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []
            
            # 内存清理间隔
            gc_interval = 10
            
            ############### Lanczos
            for i in range(iter):
                self.model.zero_grad()
                w_prime = [torch.zeros(p.size()).to(device) for p in self.params]
                if i == 0:
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item() if isinstance(alpha, torch.Tensor) else float(alpha))
                    w = group_add(w_prime, v, alpha=-int(alpha.cpu().item() if isinstance(alpha, torch.Tensor) else float(alpha)))
                    w_list.append(w)
                else:
                    beta_tensor = group_product(w, w)
                    beta = torch.sqrt(beta_tensor if isinstance(beta_tensor, torch.Tensor) else torch.tensor(beta_tensor))
                    beta_list.append(beta.cpu().item())
                    if beta_list[-1] != 0.:
                        # We should re-orth it
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    else:
                        # generate a new vector
                        w = [torch.randn(p.size()).to(device) for p in self.params]
                        v = orthnormal(w, v_list)
                        v_list.append(v)
                    if self.full_dataset:
                        _, w_prime = self.dataloader_hv_product(v)
                    else:
                        w_prime = hessian_vector_product(
                            self.gradsH, self.params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item() if isinstance(alpha, torch.Tensor) else float(alpha))
                    w_tmp = group_add(w_prime, v, alpha=-int(alpha.cpu().item() if isinstance(alpha, torch.Tensor) else float(alpha)))
                    w = group_add(w_tmp, v_list[-2], alpha=-int(beta.cpu().item()))
                
                # 定期清理 GPU 缓存
                if (i + 1) % gc_interval == 0:
                    torch.cuda.empty_cache()

            # 创建tridiagonal矩阵T
            T = torch.zeros(iter, iter).to(device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]

            # 检查是否有NaN或Inf值
            if torch.isnan(T).any() or torch.isinf(T).any():
                print("Warning: NaN or Inf values detected in matrix T")
                # 用零替换NaN和Inf值
                T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                # 使用torch.linalg.eig替代torch.eig
                eigvals, eigvecs = torch.linalg.eig(T)
                eigen_list = eigvals.real  # 取实部
                weight_list = eigvecs[0, :].real ** 2  # 取实部并平方

                eigen_list_full.append(list(eigen_list.cpu().numpy()))
                weight_list_full.append(list(weight_list.cpu().numpy()))
            except RuntimeError as e:
                print(f"Error in eigendecomposition: {e}")
                # 返回空列表作为备用
                eigen_list_full.append([0.0] * iter)
                weight_list_full.append([1.0 / iter] * iter)

        return eigen_list_full, weight_list_full

