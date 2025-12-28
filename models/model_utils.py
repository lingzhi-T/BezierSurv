
from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.loss_func import NLLSurvLoss
class BilinearFusion(nn.Module):
    r"""
    Late Fusion Block using Bilinear Pooling

    args:
        skip (int): Whether to input features at the end of the layer
        use_bilinear (bool): Whether to use bilinear pooling during information gating
        gate1 (bool): Whether to apply gating to modality 1
        gate2 (bool): Whether to apply gating to modality 2
        dim1 (int): Feature mapping dimension for modality 1
        dim2 (int): Feature mapping dimension for modality 2
        scale_dim1 (int): Scalar value to reduce modality 1 before the linear layer
        scale_dim2 (int): Scalar value to reduce modality 2 before the linear layer
        mmhid (int): Feature mapping dimension after multimodal fusion
        dropout_rate (float): Dropout rate
    """
    def __init__(self, skip=0, use_bilinear=0, gate1=1, gate2=1, dim1=128, dim2=128, scale_dim1=1, scale_dim2=1, mmhid=256, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1_og+dim2_og if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), 256), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(256+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))

    def forward(self, vec1, vec2):
        
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            h1 = self.linear_h1(vec1)
            o1 = self.linear_o1(h1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            h2 = self.linear_h2(vec2)
            o2 = self.linear_o2(h2)

        ### Fusion
        o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, vec1, vec2), 1)
        out = self.encoder2(out)
        return out


def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))


def Reg_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block (Linear + ReLU + Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Dropout(p=dropout, inplace=False))


class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


def init_max_weights(module):
    r"""
    Initialize Weights function.

    args:
        modules (torch.nn.Module): Initalize weight using normal distribution
    """
    import math
    import torch.nn as nn
    
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()



class IBPathBlock(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, bottleneck_dim=256, out_dim=1,return_z=False,seed=None,use_mean_in_testing=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.fc_mu = nn.Linear(hidden_dim, bottleneck_dim)
        self.fc_std = nn.Linear(hidden_dim, bottleneck_dim)
        self.decoder = nn.Linear(bottleneck_dim, out_dim)
        self.return_z =return_z
        self.seed = seed
        self.use_mean_in_testing =use_mean_in_testing

    # def reparameterize(self, mu, std):
    #     eps = torch.randn_like(std)
    #     return mu + std * eps

    def reparameterize(self, mu, std, training=True, seed=None):
        """
        mu : [batch_size, z_dim]
        std : [batch_size, z_dim]
        training : bool, whether we are in training mode or testing mode
        seed : int, random seed for testing phase (if training=False)
        """
        # If we are in training, generate eps from standard normal (mean=0, std=1)
        if training:
            # Sample epsilon from normal distribution with mean=0 and std=1

            # eps = torch.normal(torch.zeros_like(std), torch.ones_like(std)).cuda()
            eps = torch.randn_like(std)
        else:
            # In testing phase, use a fixed random seed to ensure consistency
            generator = torch.cuda.manual_seed(seed)  # Set the seed for reproducibility
            eps = torch.normal(torch.zeros_like(std), torch.ones_like(std), generator=generator).cuda()

        return mu + std * eps

    def forward(self, x, y_disc, event_time, censor, beta=1.0,is_training=True):
        B = y_disc.size(0)
        h = self.encoder(x)               # [B, H]
        mu = self.fc_mu(h)                # [B, D]
        std = F.softplus(self.fc_std(h) - 5) + 1e-6  # [B, D]
        if is_training:
            z = self.reparameterize(mu, std)
        else:
            ## 
            if self.use_mean_in_testing:
                z = mu  # 推理阶段使用均值
            else:
                z = self.reparameterize(mu, std,training=is_training,seed=self.seed)

        # z = self.reparameterize(mu, std)  # [B, D]
        pred = self.decoder(z)            # [B, 1] or [B, C] if multi-bin

        # 计算重建损失（生存预测）
        recon_list = []
        for i in range(B):
            if B == 1:
                p_i = pred if len(pred.shape) > 1 else pred.unsqueeze(0)
            else:
                p_i = pred[i].unsqueeze(0)
            y_i = y_disc[i].unsqueeze(0)
            t_i = event_time[i].unsqueeze(0)
            c_i = censor[i].unsqueeze(0)
            loss_i = NLLSurvLoss(reduction='mean')(h=p_i, y=y_i, t=t_i, c=c_i)
            recon_list.append(loss_i)
        recon = torch.stack(recon_list)  # [B]

        # KL divergence
        kl = 0.5 * torch.mean(mu.pow(2) + std.pow(2) - 2 * torch.log(std) - 1)
        total_loss = recon + beta * kl  # [B]

        if self.return_z:
            return total_loss, z, mu, std  # 用于模态间对齐
        else:
            return total_loss

## 深度证据回归所用的模块
class DenseNormalGamma(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DenseNormalGamma, self).__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        self.dense = nn.Linear(self.in_dim, 4 * self.out_dim)

    def evidence(self, x):
        return F.softplus(x)

    def forward(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = output.chunk(4, dim=-1)

        v = self.evidence(logv)
        alpha = self.evidence(logalpha) + 1
        beta = self.evidence(logbeta)

        aleatoric = beta / (alpha - 1)
        epistemic = beta / v * (alpha - 1)

        return mu, v, alpha, beta, aleatoric, epistemic
    


class MultiheadAttention(nn.Module):
    def __init__(self,
                 q_dim = 256,
                 k_dim = 256,
                 v_dim = 256,
                 embed_dim = 256,
                 out_dim = 256,
                 n_head = 4,
                 dropout=0.1,
                 temperature = 1
                 ):
        super(MultiheadAttention, self).__init__()
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = self.embed_dim//self.n_head
        self.temperature = temperature


        self.w_q = nn.Linear(self.q_dim, embed_dim)
        self.w_k = nn.Linear(self.k_dim, embed_dim)
        self.w_v = nn.Linear(self.v_dim, embed_dim)

        self.scale = (self.embed_dim//self.n_head) ** -0.5

        self.attn_dropout = nn.Dropout(self.dropout)
        self.proj_dropout = nn.Dropout(self.dropout)

        # self.layerNorm1 = nn.LayerNorm(out_dim)
        # self.layerNorm2 = nn.LayerNorm(out_dim)

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout)
        )
        
        # self.feedForward = nn.Sequential(
        #     nn.Linear(out_dim, embed_dim),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(embed_dim, out_dim)
        # )

    def forward(self, q, k, v, return_attn = False):
        q_raw = q
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        batch_size = q.shape[0] # B
        q = q.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)
        k = k.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)
        v = v.view(batch_size, -1, self.n_head, self.head_dim).transpose(1,2)

        attention_score = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention_score = F.softmax(attention_score / self.temperature, dim = -1)

        attention_score = self.attn_dropout(attention_score)

        x = torch.matmul(attention_score, v)

        attention_score = attention_score.sum(dim = 1)/self.n_head
        
        attn_out = x.transpose(1,2).contiguous().view(batch_size, -1, self.embed_dim)

        attn_out = self.out_proj(attn_out)

        attn_out = self.proj_dropout(attn_out)

        # attn_out = attn_out + q_raw

        # attn_out = self.layerNorm1(attn_out)

        # out = self.feedForward(attn_out)

        # out = self.layerNorm2(out + attn_out)

        # out = self.dropout(out)
        if return_attn:
            return attn_out, attention_score
        else:
            return attn_out
        # return out, attention_score


def initialize_weights(module):
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.xavier_normal_(m.weight)
			m.bias.data.zero_()
		
		elif isinstance(m, nn.BatchNorm1d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

class Attn_Net(nn.Module):

    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        
        self.module = nn.Sequential(*self.module)
    
    def forward(self, x):
        return self.module(x), x # N x n_classes


import opt_einsum as oe
from einops import rearrange, repeat
import math
_c2r = torch.view_as_real
_r2c = torch.view_as_complex

class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """ X: (batch, dim, lengths...) """
        if self.training:
            if not self.transposed:
                X = rearrange(X, 'b d ... -> b ... d')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed:
                X = rearrange(X, 'b ... d -> b d ...')
            return X
        return X


class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(_c2r(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = _r2c(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):  # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L)  # (H L)
        u_f = torch.fft.rfft(u.to(torch.float32), n=2*L)  # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L]  # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y


class S4Model_feature(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act, survival = False):
        super(S4Model_feature, self).__init__()
        self.n_classes = n_classes
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
            print("dropout: ", dropout)
        self._fc1 = nn.Sequential(*self._fc1)
        self.s4_block = nn.Sequential(nn.LayerNorm(512),
                                      S4D(d_model=512, d_state=32, transposed=False))

        self.classifier = nn.Linear(512, self.n_classes)
        self.survival = survival
    def forward(self, x):
        # x = x.unsqueeze(0)
        # print(x.shape)
        x = self._fc1(x)
        x = self.s4_block(x)
        x = torch.max(x, axis=1).values
        # print(x.shape)
        logits = self.classifier(x)
        # Y_prob = F.softmax(logits, dim=1)
        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # A_raw = None
        # results_dict = None
        # if self.survival:
        #     Y_hat = torch.topk(logits, 1, dim = 1)[1]
        #     hazards = torch.sigmoid(logits)
        #     S = torch.cumprod(1 - hazards, dim=1)
        #     return hazards, S, Y_hat, None, None
        # return logits, Y_prob, Y_hat, A_raw, results_dict
        return logits


    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1 = self._fc1.to(device)
        self.s4_block  = self.s4_block .to(device)
        self.classifier = self.classifier.to(device)

# q = torch.randn(4, 1, 256)  # Q with shape [3, 1, 256]
# k = torch.randn(1, 1, 4096, 256)  # K with shape [1, 1, 4096, 256]
# v = torch.randn(1, 1, 4096, 256)  # V with shape [1, 1, 4096, 256]

# multihead_attention = MultiheadAttention(q_dim=256, k_dim=256, v_dim=256, embed_dim=256, out_dim=256, n_head=4, dropout=0.1)
# output, attention_scores = multihead_attention(q, k, v, return_attn=True)
# print(output.shape)  # Expected output shape: [1, 3, 256]

"""
MambaMIL
"""
# import torch
# import torch.nn as nn
# from mamba.mamba_ssm import SRMamba
# from mamba.mamba_ssm import BiMamba
# from mamba.mamba_ssm import Mamba
# import torch.nn.functional as F


# def initialize_weights(module):
#     for m in module.modules():
#         if isinstance(m, nn.Linear):
#             nn.init.xavier_normal_(m.weight)
#             if m.bias is not None:
#                 m.bias.data.zero_()
#         if isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)


# class MambaMIL(nn.Module):
#     def __init__(self, in_dim, n_classes, dropout, act, survival = False, layer=2, rate=10, type="SRMamba"):
#         super(MambaMIL, self).__init__()
#         self._fc1 = [nn.Linear(in_dim, 512)]
#         if act.lower() == 'relu':
#             self._fc1 += [nn.ReLU()]
#         elif act.lower() == 'gelu':
#             self._fc1 += [nn.GELU()]
#         if dropout:
#             self._fc1 += [nn.Dropout(dropout)]

#         self._fc1 = nn.Sequential(*self._fc1)
#         self.norm = nn.LayerNorm(512)
#         self.layers = nn.ModuleList()
#         self.survival = survival

#         if type == "SRMamba":
#             for _ in range(layer):
#                 self.layers.append(
#                     nn.Sequential(
#                         nn.LayerNorm(512),
#                         SRMamba(
#                             d_model=512,
#                             d_state=16,  
#                             d_conv=4,    
#                             expand=2,
#                         ),
#                         )
#                 )
#         elif type == "Mamba":
#             for _ in range(layer):
#                 self.layers.append(
#                     nn.Sequential(
#                         nn.LayerNorm(512),
#                         Mamba(
#                             d_model=512,
#                             d_state=16,  
#                             d_conv=4,    
#                             expand=2,
#                         ),
#                         )
#                 )
#         elif type == "BiMamba":
#             for _ in range(layer):
#                 self.layers.append(
#                     nn.Sequential(
#                         nn.LayerNorm(512),
#                         BiMamba(
#                             d_model=512,
#                             d_state=16,  
#                             d_conv=4,    
#                             expand=2,
#                         ),
#                         )
#                 )
#         else:
#             raise NotImplementedError("Mamba [{}] is not implemented".format(type))

#         self.n_classes = n_classes
#         self.classifier = nn.Linear(512, self.n_classes)
#         self.attention = nn.Sequential(
#             nn.Linear(512, 128),
#             nn.Tanh(),
#             nn.Linear(128, 1)
#         )
#         self.rate = rate
#         self.type = type

#         self.apply(initialize_weights)

#     def forward(self, x):
#         if len(x.shape) == 2:
#             x = x.expand(1, -1, -1)
#         h = x.float()  # [B, n, 1024]
        
#         h = self._fc1(h)  # [B, n, 256]

#         if self.type == "SRMamba":
#             for layer in self.layers:
#                 h_ = h
#                 h = layer[0](h)
#                 h = layer[1](h, rate=self.rate)
#                 h = h + h_
#         elif self.type == "Mamba" or self.type == "BiMamba":
#             for layer in self.layers:
#                 h_ = h
#                 h = layer[0](h)
#                 h = layer[1](h)
#                 h = h + h_

#         h = self.norm(h)
#         A = self.attention(h) # [B, n, K]
#         A = torch.transpose(A, 1, 2)
#         A = F.softmax(A, dim=-1) # [B, K, n]
#         h = torch.bmm(A, h) # [B, K, 512]
#         h = h.squeeze(0)

#         logits = self.classifier(h)  # [B, n_classes]
#         # Y_prob = F.softmax(logits, dim=1)
#         # Y_hat = torch.topk(logits, 1, dim=1)[1]
#         # A_raw = None
#         # results_dict = None
#         # if self.survival:
#         #     Y_hat = torch.topk(logits, 1, dim = 1)[1]
#         #     hazards = torch.sigmoid(logits)
#         #     S = torch.cumprod(1 - hazards, dim=1)
#         #     return hazards, S, Y_hat, None, None
#         # return logits, Y_prob, Y_hat, A_raw, results_dict
#         return logits
    
#     def relocate(self):
#         device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self._fc1 = self._fc1.to(device)
#         self.layers  = self.layers.to(device)
        
#         self.attention = self.attention.to(device)
#         self.norm = self.norm.to(device)
#         self.classifier = self.classifier.to(device)

        








