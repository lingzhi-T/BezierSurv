## 带memory 消融实验版本

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import MultiheadAttention
from models.model_utils import *
from nystrom_attention import NystromAttention



## 关于besier 的消融实验
## 设置一个超参数控制是或否控制测试过程的随机性
class BezierSurv(nn.Module):
    def __init__(self, 
                 omic_input_dim, 
                 fusion='concat', 
                 n_classes=4,
                 model_size_path: str='small', 
                 model_size_geno: str='small', 
                 mil_model_type='TransMIL',
                 geno_mlp_type='SNN',
                 memory_size=32,
                 n_ctrl_points=3,
                 dropout=0.1,
             
                 use_ib_path=True,
                 use_ib_geno=True,
                 use_ib_fusion=True,
                 use_kl_align=True,
                 use_align_loss=True,
                 use_sim_loss=True,
                 use_bezier_gmm=True,
                # 新增三条 β
                 beta_path= 1.0,
                 beta_geno= 1.0,
                 beta_fusion= 1.0,
                 prototype_mode='bezier_gmm',
                 seed=1,
                 use_mean_in_testing= False  ## 测试过程是采用均值还是固定随机数的重参数化,默认为关闭，即测试过程和训练过程一致
                 ):

        super(BezierSurv, self).__init__()
        self.fusion = fusion
        self.geno_input_dim = omic_input_dim
        self.n_classes = n_classes
        self.use_ib_path = use_ib_path
        self.use_ib_geno = use_ib_geno
        self.use_ib_fusion = use_ib_fusion
        self.use_kl_align = use_kl_align
        self.use_align_loss = use_align_loss
        self.use_sim_loss = use_sim_loss
        self.use_bezier_gmm = use_bezier_gmm

        self.size_dict_path = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_geno = {'small': [1024, 256], 'big': [1024, 1024, 1024, 256]}

        ### Define Prototype Bank
        self.memory_size = memory_size
        self.memory_dim = 256
        '''
        # 均值端点（起始类、终止类）
        self.path_mu_endpoints = nn.Parameter(torch.randn(2, self.memory_size, self.memory_dim) * 0.1)  # [2, K, D]
        init_pi = torch.randn(self.memory_size) * 0.01
        init_pi_all = init_pi.unsqueeze(0).repeat(self.n_classes, 1)  # [C, K]
        self.path_proto_pi = nn.Parameter(init_pi_all)
        self.path_bezier_ctrl = nn.Parameter(torch.randn(n_ctrl_points,  self.memory_dim) * 0.1)  # [n_ctrl, D]
        self.path_proto_std = nn.Parameter(torch.randn(self.n_classes, self.memory_size, self.memory_dim) * 0.1)  # [C, K, D]

        self.geno_mu_endpoints = nn.Parameter(torch.randn(2, self.memory_size, self.memory_dim) * 0.1)  # [2, K, D]
        init_pi = torch.randn(self.memory_size) * 0.01
        init_pi_all = init_pi.unsqueeze(0).repeat(self.n_classes, 1)
        self.geno_proto_pi = nn.Parameter(init_pi_all)
        self.geno_bezier_ctrl = nn.Parameter(torch.randn(n_ctrl_points,  self.memory_dim) * 0.1)  # [n_ctrl, D]
        self.geno_proto_std = nn.Parameter(torch.randn(self.n_classes, self.memory_size, self.memory_dim) * 0.1)  # [C, K, D]
        '''

        ## 用于网络里进行控制 选择 对应的memory bank 生成方式
        self.prototype_mode = prototype_mode
        self.use_mean_in_testing = use_mean_in_testing  ## 推理过程是否使用均值
        self.seed = seed
        if prototype_mode == 'memory_bank':
            self.path_prototype_bank = nn.Parameter(
                torch.empty(self.n_classes, self.memory_size, self.memory_dim)
            )
            nn.init.xavier_uniform_(self.path_prototype_bank, gain=1.0)
            self.geno_prototype_bank = nn.Parameter(
                torch.empty(self.n_classes, self.memory_size, self.memory_dim)
            )
            nn.init.xavier_uniform_(self.geno_prototype_bank, gain=1.0)

        # 如果用 SingleGaussian
        elif prototype_mode == 'single_gaussian':
            # 只学一个 mu
            # self.path_mu_singlegauss = nn.Parameter(torch.randn(self.n_classes, self.memory_dim) * 0.1)
            # self.geno_mu_singlegauss = nn.Parameter(torch.randn(self.n_classes, self.memory_dim) * 0.1)

            self.path_proto_mu = nn.Parameter(torch.empty(self.n_classes, 1, self.memory_dim))
            self.path_proto_std = nn.Parameter(torch.empty(self.n_classes, 1, self.memory_dim))
            
            # 基因模态的原型均值和标准差
            self.geno_proto_mu = nn.Parameter(torch.empty(self.n_classes, 1, self.memory_dim))
            self.geno_proto_std = nn.Parameter(torch.empty(self.n_classes, 1, self.memory_dim))

            #初始化
            torch.nn.init.xavier_uniform_(self.path_proto_mu)
            torch.nn.init.xavier_uniform_(self.geno_proto_mu)
            torch.nn.init.constant_(self.path_proto_std, 0.1)  # 小正值避免过小方差
            torch.nn.init.constant_(self.geno_proto_std, 0.1)


        # 如果用 GMM
        elif prototype_mode == 'gmm':
            self.path_proto_mu = nn.Parameter(torch.empty(self.n_classes, self.memory_size , self.memory_dim))  # [C, K, D]
            self.path_proto_std = nn.Parameter(torch.empty(self.n_classes, self.memory_size , self.memory_dim)) # [C, K, D]
            self.path_proto_pi = nn.Parameter(torch.ones(self.n_classes, self.memory_size ))                     # [C, K]

            # 模态 2: 基因模态 GMM proxy
            self.geno_proto_mu = nn.Parameter(torch.empty(self.n_classes, self.memory_size , self.memory_dim))  # [C, K, D]
            self.geno_proto_std = nn.Parameter(torch.empty(self.n_classes, self.memory_size, self.memory_dim)) # [C, K, D]
            self.geno_proto_pi = nn.Parameter(torch.ones(self.n_classes, self.memory_size))                    # [C, K]
            # ========== 均值 ==========
            torch.nn.init.xavier_uniform_(self.path_proto_mu)
            torch.nn.init.xavier_uniform_(self.geno_proto_mu)

            # ========== 方差 ==========
            torch.nn.init.constant_(self.path_proto_std, 0.1)
            torch.nn.init.constant_(self.geno_proto_std, 0.1)

            # ========== 权重 logits（再 softmax）==========
            torch.nn.init.normal_(self.path_proto_pi, mean=0.0, std=0.01)
            torch.nn.init.normal_(self.geno_proto_pi, mean=0.0, std=0.01)

        # 如果用 BezierGMM
        elif prototype_mode == 'bezier_gmm':
            self.path_mu_endpoints = nn.Parameter(torch.randn(2, self.memory_size, self.memory_dim) * 0.1)  # [2, K, D]
            init_pi = torch.randn(self.memory_size) * 0.01
            init_pi_all = init_pi.unsqueeze(0).repeat(self.n_classes, 1)  # [C, K]
            self.path_proto_pi = nn.Parameter(init_pi_all)
            self.path_bezier_ctrl = nn.Parameter(torch.randn(n_ctrl_points,  self.memory_dim) * 0.1)  # [n_ctrl, D]
            self.path_proto_std = nn.Parameter(torch.randn(self.n_classes, self.memory_size, self.memory_dim) * 0.1)  # [C, K, D]

            self.geno_mu_endpoints = nn.Parameter(torch.randn(2, self.memory_size, self.memory_dim) * 0.1)  # [2, K, D]
            init_pi = torch.randn(self.memory_size) * 0.01
            init_pi_all = init_pi.unsqueeze(0).repeat(self.n_classes, 1)
            self.geno_proto_pi = nn.Parameter(init_pi_all)
            self.geno_bezier_ctrl = nn.Parameter(torch.randn(n_ctrl_points,  self.memory_dim) * 0.1)  # [n_ctrl, D]
            self.geno_proto_std = nn.Parameter(torch.randn(self.n_classes, self.memory_size, self.memory_dim) * 0.1)  # [C, K, D]
            print("ceshi")
        else:
            raise ValueError(f"Unsupported prototype_mode: {prototype_mode}")

        # 保存 β
        # self.beta_path = nn.Parameter(torch.tensor(beta_path))
        # self.beta_geno = nn.Parameter(torch.tensor(beta_geno))
        # self.beta_fusion = nn.Parameter(torch.tensor(beta_fusion))

        self.beta_path = 1
        self.beta_geno = 1
        self.beta_fusion =1
        

        self.path_ib_block = IBPathBlock(input_dim=self.memory_dim, bottleneck_dim=256, out_dim=self.n_classes, return_z=True,seed=seed,use_mean_in_testing=use_mean_in_testing)
        self.geno_ib_block = IBPathBlock(input_dim=self.memory_dim, bottleneck_dim=256, out_dim=self.n_classes, return_z=True,seed=seed,use_mean_in_testing=use_mean_in_testing)
        ## 被抛弃
        # self.fusion_ib_block = IBPathBlock(input_dim=self.memory_dim, bottleneck_dim=256, out_dim=self.n_classes, return_z=True,seed=seed,use_mean_in_testing=use_mean_in_testing)
        
        ### Pathology FC
        size = self.size_dict_path[model_size_path]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.LayerNorm(normalized_shape=size[1]))
        fc.append(nn.Dropout(dropout))
        self.path_proj = nn.Sequential(*fc)
        self.path_attn_net = pathMIL(model_type=mil_model_type, input_dim=size[1], dropout=dropout)

        ### Genomic SNN
        hidden = self.size_dict_geno[model_size_geno]
        if geno_mlp_type == 'SNN':
            geno_snn = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                geno_snn.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout))
            self.geno_snn = nn.Sequential(*geno_snn)
        else:
            self.geno_snn = nn.Sequential(
                nn.Linear(omic_input_dim, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout))

        ### Multihead Attention
        self.path_intra_read_attn = MultiheadAttention(q_dim=self.size_dict_geno[model_size_geno][-1], k_dim=self.memory_dim, 
                                                       v_dim=self.memory_dim, embed_dim=size[1], out_dim=size[1], 
                                                       n_head=4, dropout=dropout, temperature=0.5)

        self.geno_intra_read_attn = MultiheadAttention(q_dim=size[1], k_dim=self.memory_dim, 
                                                       v_dim=self.memory_dim, embed_dim=size[1], out_dim=size[1], 
                                                       n_head=4, dropout=dropout, temperature=0.5)

        ### Fusion Layer
        if self.fusion == 'concat' or self.fusion == 'ibweighted':
            self.mm = nn.Sequential(*[nn.Linear(size[1]*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU()])

        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_geno = kwargs['x_omic']
        label = kwargs['label']
        censor = kwargs['censor']
        is_training = kwargs['training']
        input_modality = kwargs['input_modality']
        survival_time = kwargs["event_time"]

        batch_size = x_path.shape[0] if x_path is not None else x_geno.shape[0]

        if x_path is not None:
            h_path = self.path_proj(x_path)
            h_path = self.path_attn_net(h_path)
        if x_geno is not None:
            h_geno = self.geno_snn(x_geno).squeeze(1)

        if self.use_ib_path and (input_modality in ['path', 'path_and_geno']):
            ib_path_loss, h_path, mu_path, std_path = self.path_ib_block(h_path, label, survival_time, censor,  beta=self.beta_path, is_training=is_training)
        else:
            ib_path_loss, mu_path, std_path = torch.tensor(0.0, device=h_path.device).unsqueeze(dim=0), torch.zeros_like(h_path), torch.ones_like(h_path)

        if self.use_ib_geno and (input_modality in ['geno', 'path_and_geno']):
            ib_geno_loss, h_geno, mu_geno, std_geno = self.geno_ib_block(h_geno, label, survival_time, censor, beta=self.beta_geno, is_training=is_training)
        else:
            ib_geno_loss, mu_geno, std_geno = torch.tensor(0.0, device=h_geno.device).unsqueeze(dim=0), torch.zeros_like(h_geno), torch.ones_like(h_geno)

        sim_loss = torch.tensor(0.0, device=h_path.device)
        if is_training and self.use_sim_loss:
            path_sim_loss = 0.
            geno_sim_loss = 0.
            if input_modality in ['path', 'path_and_geno']:
                
                ## 原始bezier_gmm版本
                h_path_norm = F.normalize(h_path)
                # path_mu_all = generate_bezier_global_path(self.path_mu_endpoints, self.path_bezier_ctrl, self.path_proto_pi, self.n_classes, bezier_curve)
                # path_proto_bank = gmm_sample(path_mu_all, self.path_proto_std, self.path_proto_pi, sample_times=self.memory_size)
                # path_sim = gmm_contrastive_similarity(h_path_norm, path_proto_bank, temperature=0.07)
                # path_sim_loss = censor_margin_loss(sim=path_sim, label=label, censor=censor, sample_times=self.memory_size)

                if self.prototype_mode == 'memory_bank':
                    path_proto_bank = F.normalize(self.path_prototype_bank.reshape(
                    self.n_classes*self.memory_size, self.memory_dim)) #[n_classes*size, D]
                elif self.prototype_mode == 'single_gaussian':
                    # self.path_mu_singlegauss: [C, D]，相当于每类一个 center
                    # 这里我们把每类均值当作一个“样本”，sample_times = 1
                    # flat = self.path_mu_singlegauss.view(self.n_classes, self.memory_dim)  # [C, D]
                    # path_proto_bank = flat.unsqueeze(0).expand(batch_size, -1, -1)         # [B, C, D]
                    proto_kcd_path = get_proto_test(self.path_proto_mu, self.path_proto_std,
                                       sample_times=self.memory_size)  # [K, C, D]
                    # 需要将它转换为 [C*K, D]，然后扩展到 batch
                    #   当前 proto_kcd 形状是 [K, C, D]，我们先 permute 成 [C, K, D]
                    proto_kcd_path = proto_kcd_path.permute(1, 0, 2)      # [C, K, D]
                    path_proto_bank  = proto_kcd_path.contiguous().view(self.n_classes * self.memory_size, self.memory_dim)  # [C*K, D]
                    # path_proto_bank = flat_ckd.unsqueeze(0).expand(batch_size, -1, -1)  # [B, C*K, D]

                elif self.prototype_mode == 'gmm':
                    # self.path_mu_gmm: [C, K, D], self.path_std_gmm: [C, K, D], self.path_logits_pi: [C, K]
                    # 按照 π 抽样 sample_times = self.memory_size 次
                    path_proto_bank =gmm_sample_v2(self.path_proto_mu, self.path_proto_std, self.path_proto_pi, 
                         sample_times=self.memory_size)

                elif self.prototype_mode == 'bezier_gmm':
                    path_mu_all = generate_bezier_global_path(self.path_mu_endpoints, self.path_bezier_ctrl, self.path_proto_pi, self.n_classes, bezier_curve)
                    path_proto_bank = gmm_sample(path_mu_all, self.path_proto_std, self.path_proto_pi, sample_times=self.memory_size)
                else:
                    raise ValueError(f"Unsupported prototype_mode: {self.prototype_mode}")
                
                path_sim = gmm_contrastive_similarity(h_path_norm, path_proto_bank, temperature=0.07)
                path_sim_loss = censor_margin_loss(sim=path_sim, label=label, censor=censor, sample_times=self.memory_size)

            if input_modality in ['geno', 'path_and_geno']:
                # h_geno_norm = F.normalize(h_geno)
                # geno_mu_all = generate_bezier_global_path(self.geno_mu_endpoints, self.geno_bezier_ctrl, self.geno_proto_pi, self.n_classes, bezier_curve)
                # geno_sim = gmm_sample_contrastive(h_geno_norm, geno_mu_all, self.geno_proto_std, self.geno_proto_pi, self.memory_size)
                # sim_loss += censor_margin_loss(sim=geno_sim, label=label, censor=censor, sample_times=self.memory_size)
                ## 切换一个版本
                h_geno_norm = F.normalize(h_geno)
                if self.prototype_mode == 'memory_bank':
                    # 直接把 learnable 的 memory_bank reshape 并扩展到 batch 维度
                    # self.path_prototype_bank: [C, K, D] → flat: [C*K, D]
                    # flat = self.path_prototype_bank.view(self.n_classes * self.memory_size, self.memory_dim)
                    geno_proto_bank = F.normalize(self.geno_prototype_bank.reshape(
                    self.n_classes*self.memory_size, self.memory_dim)) #[n_classes*size, D]
                    # path_proto_bank = flat.unsqueeze(0).expand(batch_size, -1, -1)  # [B, C*K, D]
                elif self.prototype_mode == 'bezier_gmm':
                    
                    geno_mu_all = generate_bezier_global_path(self.geno_mu_endpoints, self.geno_bezier_ctrl, self.geno_proto_pi, self.n_classes, bezier_curve)
                    geno_proto_bank = gmm_sample(geno_mu_all, self.geno_proto_std, self.geno_proto_pi, self.memory_size)

                elif self.prototype_mode == 'single_gaussian':
                    # self.path_mu_singlegauss: [C, D]，相当于每类一个 center
                    # 这里我们把每类均值当作一个“样本”，sample_times = 1
                    # flat = self.path_mu_singlegauss.view(self.n_classes, self.memory_dim)  # [C, D]
                    # path_proto_bank = flat.unsqueeze(0).expand(batch_size, -1, -1)         # [B, C, D]

                    proto_kcd_geno = get_proto_test(self.geno_proto_mu, self.geno_proto_std,
                                       sample_times=self.memory_size)  # [K, C, D]

                    # 需要将它转换为 [C*K, D]，然后扩展到 batch
                    #   当前 proto_kcd 形状是 [K, C, D]，我们先 permute 成 [C, K, D]
                    proto_kcd_geno = proto_kcd_geno.permute(1, 0, 2)      # [C, K, D]
                    geno_proto_bank  = proto_kcd_geno.contiguous().view(self.n_classes * self.memory_size, self.memory_dim)  # [C*K, D]
                    # path_proto_bank = flat_ckd.unsqueeze(0).expand(batch_size, -1, -1)  # [B, C*K, D]
                elif self.prototype_mode == 'gmm':
                    # self.path_mu_gmm: [C, K, D], self.path_std_gmm: [C, K, D], self.path_logits_pi: [C, K]
                    # 按照 π 抽样 sample_times = self.memory_size 次
                    geno_proto_bank =gmm_sample_v2(self.geno_proto_mu, self.geno_proto_std, self.geno_proto_pi, 
                         sample_times=self.memory_size)

                geno_sim = gmm_contrastive_similarity(h_geno_norm, geno_proto_bank, temperature=0.07)
                # path_sim = gmm_sample_contrastive(h_path_norm, path_mu_all, self.path_proto_std, self.path_proto_pi, self.memory_size)
                geno_sim_loss = censor_margin_loss(sim=geno_sim, label=label, censor=censor, sample_times=self.memory_size)

            sim_loss = path_sim_loss + geno_sim_loss

        if input_modality in ['geno', 'path_and_geno']:
            if self.prototype_mode == 'bezier_gmm':
                if is_training:
                    if not(self.use_sim_loss):
                        path_mu_all = generate_bezier_global_path(self.path_mu_endpoints, self.path_bezier_ctrl, self.path_proto_pi, self.n_classes, bezier_curve)
                    path_proto_bank = gmm_sample(path_mu_all, self.path_proto_std, self.path_proto_pi, sample_times=self.memory_size)
                    path_proto_bank_flat = path_proto_bank.reshape(self.n_classes * self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1)
                # else:
                #     ## 测试一下是否随机性一致
                #     path_proto_mu = generate_bezier_global_path(self.path_mu_endpoints, self.path_bezier_ctrl, self.path_proto_pi, self.n_classes, bezier_curve)
                #     path_proto_bank = gmm_sample(path_proto_mu, self.path_proto_std, self.path_proto_pi, sample_times=self.memory_size)
                #     path_proto_bank_flat = path_proto_bank.reshape(self.n_classes * self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1)
                else:
                    if self.use_mean_in_testing:
                        path_proto_mu = generate_bezier_global_path(self.path_mu_endpoints, self.path_bezier_ctrl, self.path_proto_pi, self.n_classes, bezier_curve)
                        path_proto_bank_flat = path_proto_mu.reshape(self.n_classes, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1)
                    else:
                        path_proto_mu = generate_bezier_global_path(self.path_mu_endpoints, self.path_bezier_ctrl, self.path_proto_pi, self.n_classes, bezier_curve)
                        path_proto_bank = gmm_sample(path_proto_mu, self.path_proto_std, self.path_proto_pi, sample_times=self.memory_size,is_training=is_training,seed=self.seed)
                        path_proto_bank_flat = path_proto_bank.reshape(self.n_classes * self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1)
                

                    # path_proto_bank_flat = path_proto_mu.reshape(self.n_classes, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1)
                    
            elif self.prototype_mode == 'memory_bank':
                path_proto_bank_flat = self.path_prototype_bank.reshape(
                    self.n_classes*self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1) #[B, n_classes*size, D]
                
            elif self.prototype_mode == 'single_gaussian':
                if is_training:
                    if not(self.use_sim_loss):
                        proto_kcd_path = get_proto_test(self.path_proto_mu, self.path_proto_std,
                                       sample_times=self.memory_size)  # [K, C, D]
                    path_proto_bank_flat = proto_kcd_path.reshape(
                        self.n_classes*self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1) #[B, n_classes*size, D]
                else:
                    path_proto_mu = self.path_proto_mu.detach()  # or geno_proto_mu
                    path_prototype_bank = path_proto_mu.reshape(self.n_classes, self.memory_dim)
                    path_proto_bank_flat = path_prototype_bank.unsqueeze(0).expand(batch_size, -1, -1)

            elif self.prototype_mode == 'gmm':
                if is_training:
                    if not(self.use_sim_loss):
                        path_proto_bank =gmm_sample_v2(self.path_proto_mu, self.path_proto_std, self.path_proto_pi, 
                             sample_times=self.memory_size)
                    path_proto_bank_flat = path_proto_bank.reshape(
                        self.n_classes*self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1) #[B, n_classes*size, D]
                else:
                    # pi_prob_path = F.softmax(self.path_proto_pi, dim=-1)            # [C, K]
                    # # [C, K, 1] * [C, K, D] → [C, K, D]
                    # weighted_mu_path = pi_prob_path.unsqueeze(-1) * self.path_proto_mu    # [C, K, D]
                    # # → 对 K 求和： [C, D]
                    # path_proto_bank_flat = weighted_mu_path.sum(dim=1).expand(batch_size, -1, -1)                    # [B, C, D]

                    path_proto_bank_flat=self.path_proto_mu.reshape(
                    self.n_classes*self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1) 

            h_path_read = self.path_intra_read_attn(h_geno.unsqueeze(1), path_proto_bank_flat, path_proto_bank_flat).squeeze(1)

        if input_modality in ['path', 'path_and_geno']:
            if self.prototype_mode == 'bezier_gmm':
                if is_training:
                    if not(self.use_sim_loss):
                        geno_mu_all = generate_bezier_global_path(self.geno_mu_endpoints, self.geno_bezier_ctrl, self.geno_proto_pi, self.n_classes, bezier_curve)
                    geno_proto_bank = gmm_sample(geno_mu_all, self.geno_proto_std, self.geno_proto_pi, sample_times=self.memory_size)
                    geno_proto_bank_flat = geno_proto_bank.reshape(self.n_classes * self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1)
                else:
                    if self.use_mean_in_testing:
                        geno_proto_mu = generate_bezier_global_path(self.geno_mu_endpoints, self.geno_bezier_ctrl, self.geno_proto_pi, self.n_classes, bezier_curve)
                        # geno_proto_bank = gmm_sample(geno_proto_mu, self.geno_proto_std, self.geno_proto_pi, sample_times=self.memory_size)
                        # geno_proto_bank_flat = geno_proto_bank.reshape(self.n_classes * self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1)
                        geno_proto_bank_flat = geno_proto_mu.reshape(self.n_classes, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1)
                    else:
                        geno_mu_all = generate_bezier_global_path(self.geno_mu_endpoints, self.geno_bezier_ctrl, self.geno_proto_pi, self.n_classes, bezier_curve)
                        geno_proto_bank = gmm_sample(geno_mu_all, self.geno_proto_std, self.geno_proto_pi, sample_times=self.memory_size,is_training=is_training,seed=self.seed)
                        geno_proto_bank_flat = geno_proto_bank.reshape(self.n_classes * self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1)
                   

                        
            elif self.prototype_mode == 'memory_bank':
                geno_proto_bank_flat = self.geno_prototype_bank.reshape(
                    self.n_classes*self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1) #[B, n_classes*size, D]
                
            elif self.prototype_mode == 'single_gaussian':
                if is_training:
                    if not(self.use_sim_loss):
                        proto_kcd_geno = get_proto_test(self.geno_proto_mu, self.geno_proto_std,
                                       sample_times=self.memory_size)  # [K, C, D]
                    geno_proto_bank_flat = proto_kcd_geno.reshape(
                        self.n_classes*self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1) #[B, n_classes*size, D]
                else:
                    geno_proto_mu = self.geno_proto_mu.detach()  # or geno_proto_mu
                    geno_prototype_bank = geno_proto_mu.reshape(self.n_classes, self.memory_dim)
                    geno_proto_bank_flat = geno_prototype_bank.unsqueeze(0).expand(batch_size, -1, -1)

            elif self.prototype_mode == 'gmm':
                if is_training:
                    if not(self.use_sim_loss):
                        geno_proto_bank =gmm_sample_v2(self.geno_proto_mu, self.geno_proto_std, self.geno_proto_pi, 
                             sample_times=self.memory_size)
                    geno_proto_bank_flat = geno_proto_bank.reshape(
                        self.n_classes*self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1) #[B, n_classes*size, D]
                else:
                    # pi_prob_geno = F.softmax(self.geno_proto_pi, dim=-1)            # [C, K]
                    # # [C, K, 1] * [C, K, D] → [C, K, D]
                    # weighted_mu_geno = pi_prob_geno.unsqueeze(-1) * self.geno_proto_mu    # [C, K, D]
                    # # → 对 K 求和： [C, D]
                    # geno_proto_bank_flat = weighted_mu_geno.sum(dim=1).expand(batch_size, -1, -1)                    # [B, C, D]
                    geno_proto_bank_flat=self.geno_proto_mu.reshape(
                    self.n_classes*self.memory_size, self.memory_dim).unsqueeze(0).expand(batch_size, -1, -1) 
                
            h_geno_read = self.geno_intra_read_attn(h_path.unsqueeze(1), geno_proto_bank_flat, geno_proto_bank_flat).squeeze(1)

        if input_modality == 'path':
            h_path_read = h_path
            h_geno = h_geno_read
        elif input_modality == 'geno':
            h_geno_read = h_geno
            h_path = h_path_read
        elif input_modality == 'path_and_geno':
            pass
        else:
            raise NotImplementedError(f'input_modality: {input_modality} not suported')
        
        h_path_avg = (h_path + h_path_read) / 2
        h_geno_avg = (h_geno + h_geno_read) / 2

        if is_training and self.use_align_loss:
            path_loss_align = get_align_loss(h_path_read, h_path) if input_modality == 'path_and_geno' else 0.0
            geno_loss_align = get_align_loss(h_geno_read, h_geno) if input_modality == 'path_and_geno' else 0.0
            loss_align = path_loss_align + geno_loss_align
        else:
            loss_align = torch.tensor(0.0, device=h_path.device)

        if self.fusion == 'bilinear':
            h = self.mm(h_path_avg, h_geno_avg).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path_avg, h_geno_avg], dim=-1))
        elif self.fusion == 'ibweighted':
            ## 好像造成信息泄露了（不能使用信息瓶颈）
            path_weight, geno_weight = compute_feature_importance(h_path_avg, h_geno_avg)

            # 加权特征
            h_p_w = path_weight * h_path_avg  # [B, D]
            h_g_w = geno_weight * h_geno_avg  # [B, D]

            # 融合加权特征
            h = self.mm(torch.cat([h_p_w, h_g_w], dim=-1))  # [B, 2D]
            
        else:
            h = self.mm(h_path)

        if self.use_ib_fusion:
            ib_fusion_loss, h, mu_fusion, std_fusion = self.fusion_ib_block(h.squeeze(0), label, survival_time, censor, beta=self.beta_fusion, is_training=is_training)
        else:
            ib_fusion_loss, mu_fusion, std_fusion = torch.tensor(0.0, device=h.device).unsqueeze(dim=0), torch.zeros_like(h), torch.ones_like(h)

        ## 三元组的kl_algin_loss 不行
        # kl_align_loss = symmetric_tri_kl(mu_path, std_path, mu_geno, std_geno, mu_fusion, std_fusion) if self.use_kl_align else torch.tensor(0.0, device=h.device) 
        ## 尝试二元组的
  
        kl_align_loss = symmetric_kl_divergence(mu_path, std_path, mu_geno, std_geno) if self.use_kl_align else torch.tensor(0.0, device=h.device)
        
        logits = self.classifier(h)
        if is_training:
            return logits, sim_loss, loss_align, ib_path_loss, ib_geno_loss, ib_fusion_loss, kl_align_loss
        else:
            if kwargs.get('return_feature', False):
                return logits, h_path, h_geno_read, h_geno, h_geno_read
            else:
                return logits, ib_path_loss, ib_geno_loss, ib_fusion_loss


class Memory_without_reconstruction(nn.Module):
    def __init__(self, 
                 omic_input_dim, 
                 fusion='concat', 
                 n_classes=4,
                 model_size_path: str='small', 
                 model_size_geno: str='small', 
                 mil_model_type='TransMIL',
                 memory_size=16,
                 dropout=0.1):
        
        super(Memory_without_reconstruction, self).__init__()
        self.fusion = fusion
        self.geno_input_dim = omic_input_dim
        self.n_classes = n_classes
        self.size_dict_path = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_geno = {'small': [1024, 256], 'big': [1024, 1024, 1024, 256]}

        ### pathlogy FC
        size = self.size_dict_path[model_size_path]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.LayerNorm(normalized_shape = size[1]))
        fc.append(nn.Dropout(dropout))
        self.path_proj = nn.Sequential(*fc)

        self.path_attn_net = pathMIL(model_type=mil_model_type, input_dim=size[1], dropout=dropout)
        
        ### Genomic SNN
        hidden = self.size_dict_geno[model_size_geno]
        geno_snn = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            geno_snn.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=dropout))
        self.geno_snn = nn.Sequential(*geno_snn)
        
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(size[1]*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU()])
        
        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)
        
    def forward(self, **kwargs):
        # input data
        x_path = kwargs['x_path']
        x_geno = kwargs['x_omic']
        label = kwargs['label']
        censor = kwargs['censor']
        is_training = kwargs['training']
        input_modality = kwargs['input_modality']

        # pathlogy projection
        h_path = self.path_proj(x_path) #[B, n_patchs, D]
        # pathlogy attention net
        h_path = self.path_attn_net(h_path) #[B, D]

        # Genomic SNN
        h_geno = self.geno_snn(x_geno).squeeze(1) #[B, D]
        
        if input_modality == 'path':
            h_path_read = h_path
            h_geno = h_geno_read
        elif input_modality == 'geno':
            h_geno_read = h_geno
            h_path = h_path_read
        elif input_modality == 'path_and_geno':
            pass
        else:
            raise NotImplementedError
        
        h_path_avg = (h_path + h_path_read) /2
        h_geno_avg = (h_geno + h_geno_read) /2

        ### Fusion Layer
        if self.fusion == 'bilinear':
            h = self.mm(h_path_avg, h_geno_avg).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path_avg, h_geno_avg], dim=-1))
        else:
            h = self.mm(h_path)
                
        ### Survival Layer
        logits = self.classifier(h)
        
        return logits


def get_sim_loss(similarity, label, censor):
    similarity_positive_mean = []
    similarity_negative_mean = []
    for i in range(label.shape[0]):
        if censor[i] == 0:
            mask = torch.zeros_like(similarity[i], dtype=torch.bool)
            mask[label[i].item(), :] = True
            similarity_positive = torch.masked_select(similarity[i], mask).view(-1, similarity.size(-1)) #[n_pos, size]
            similarity_negative = torch.masked_select(similarity[i], ~mask).view(-1, similarity.size(-1)) #[n_neg, size]]
            similarity_positive_mean.append(torch.mean(torch.mean(similarity_positive, dim=-1), dim=-1)) # tensor
            similarity_negative_mean.append(torch.mean(torch.mean(similarity_negative, dim=-1), dim=-1)) # tensor

        else:
            if label[i] == 0:
                similarity_positive_mean.append(torch.mean(torch.mean(similarity[i], dim=-1), dim=-1)) # tensor
                similarity_negative_mean.append(torch.tensor(0, dtype=torch.float).cuda())
            else:   
                mask = torch.zeros_like(similarity[i], dtype=torch.bool)
                mask[label[i].item():, :] = True
                similarity_positive = torch.masked_select(similarity[i], mask).view(-1, similarity.size(-1)) #[n_pos, size]
                similarity_negative = torch.masked_select(similarity[i], ~mask).view(-1, similarity.size(-1)) #[n_neg, size]]
                similarity_positive_mean.append(torch.mean(torch.mean(similarity_positive, dim=-1), dim=-1)) # tensor
                similarity_negative_mean.append(torch.mean(torch.mean(similarity_negative, dim=-1), dim=-1)) # tensor

    # 将列表转换为张量并求和
    similarity_positive_mean = torch.stack(similarity_positive_mean) #[B]
    similarity_negative_mean = torch.stack(similarity_negative_mean) #[B]

    positive_mean_sum = torch.sum(similarity_positive_mean)
    negative_mean_sum = torch.sum(similarity_negative_mean)

    sim_loss = -positive_mean_sum + negative_mean_sum

    return sim_loss


def get_align_loss(read_feat, original_feat, align_fn='mse', reduction='none'):
    if align_fn == 'mse':
        loss_fn = nn.MSELoss(reduction=reduction)
    elif align_fn == 'l1':
        loss_fn = nn.L1Loss(reduction=reduction)
    else:
        raise NotImplementedError
    
    return torch.sum(torch.mean(loss_fn(read_feat, original_feat.detach()), dim=-1), dim=-1)


class pathMIL(nn.Module):
    def __init__(self, model_type = 'TransMIL', input_dim = 256, dropout=0.1):
        super(pathMIL, self).__init__()

        self.model_type = model_type

        if model_type == 'TransMIL':
            self.translayer1 = TransLayer(dim = input_dim)
            self.translayer2 = TransLayer(dim = input_dim)
            self.pos_layer = PPEG(dim = input_dim)
        elif model_type == 'ABMIL':
            self.path_gated_attn = Attn_Net_Gated(L=input_dim, D=input_dim, dropout=dropout, n_classes=1)
        elif model_type == 's4mil':
            self.S4Model_feature = S4Model_feature(in_dim=input_dim,n_classes=input_dim, act='relu',dropout=dropout)
        elif model_type == 'mambamil':
            self.MambaMIL = MambaMIL(in_dim=input_dim,n_classes=input_dim,act='relu',dropout=dropout)


    def forward(self, h_path):

        if self.model_type == 'TransMIL':
            H = h_path.shape[1]
            _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
            add_length = _H * _W - H
            h_path_sa = torch.cat([h_path, h_path[:,:add_length,:]], dim = 1) #[B, N, 512]
            # cls_token
            # B = h_path_sa.shape[0]
            # cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
            # h_path_sa = torch.cat((cls_tokens, h_path_sa), dim=1)
            # Translayer1
            h_path_sa = self.translayer1(h_path_sa) #[B, N, 256]
            # PPEG
            h_path_sa = self.pos_layer(h_path_sa, _H, _W) #[B, N, 256]
            # Translayer2
            h_path_sa = self.translayer2(h_path_sa) #[B, N, 256]
            # cls_token
            # h_path_sa = self.norm(h_path_sa)[:,0]
            h_path_sa = torch.mean(h_path, dim=1) #[B, 256]

            return h_path_sa

        elif self.model_type == 'ABMIL':
            A, h_path = self.path_gated_attn(h_path)
            A = torch.transpose(A, 2, 1)
            A = F.softmax(A, dim=-1) 
            h_path = torch.matmul(A, h_path).squeeze(1) #[B, D]
            return h_path
        elif self.model_type =='s4mil':
            h_path = self.S4Model_feature(h_path)
            return h_path
        elif self.model_type == 'mambamil':
            h_path = self.MambaMIL(h_path)
            return h_path
        else:
            raise NotImplementedError
            return 


class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//4,
            heads = 4,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape

        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)

        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        # x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        # x = x.flatten(2).transpose(1, 2)
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
    


def compute_all_log_likelihoods(z, proto_mu, proto_std, proto_pi, eps=1e-6):
    """
    z: [B, D]
    proto_mu: [C, K, D]
    proto_std: [C, K, D]
    proto_pi: [C, K]
    return: [B, C] — 每个样本对每个类别的 log-likelihood
    """
    B, D = z.shape
    C, K, _ = proto_mu.shape

    z = z[:, None, None, :]           # [B, 1, 1, D]
    mu = proto_mu[None, :, :, :]      # [1, C, K, D]
    std = proto_std[None, :, :, :]    # [1, C, K, D]
    pi = F.softmax(proto_pi, dim=-1)[None, :, :]  # [1, C, K]

    # [B, C, K, D]
    diff = z - mu
    exp_term = -0.5 * torch.sum((diff ** 2) / (std ** 2 + eps), dim=-1)           # [B, C, K]
    log_coeff = - torch.sum(torch.log(std + eps), dim=-1) - 0.5 * D * np.log(2 * np.pi)  # [B, C, K]
    log_probs = log_coeff + exp_term + torch.log(pi + eps)                        # [B, C, K]
    log_likelihoods = torch.logsumexp(log_probs, dim=-1)                          # [B, C]
    return log_likelihoods

def censor_mask(label, censor, num_classes):
    """
    更简洁实现：返回 [B, C] 的 mask
    """
    B = label.shape[0]
    device = label.device
    C = num_classes

    # 每行是 [0, 1, ..., C-1]
    class_ids = torch.arange(C, device=device).unsqueeze(0).expand(B, -1)  # [B, C]
    label_expand = label.unsqueeze(1)  # [B, 1]

    # 对于未删失样本：只允许该类别
    exact_match = (class_ids == label_expand)

    # 对于删失样本：允许所有 >= label[i] 的类别
    greater_equal = (class_ids >= label_expand)

    # censor: 0 → exact_match；1 → greater_equal
    mask = torch.where(censor.unsqueeze(1) == 0, exact_match, greater_equal).float()  # [B, C]

    return mask


def gmm_contrastive_censored_loss(z, label, censor,
                                   proto_mu, proto_std, proto_pi,
                                   temperature=0.07):
    """
    z: [B, D], label: [B], censor: [B], censor_time: [B]
    proto_mu: [C, K, D], etc.
    return: scalar loss
    """
    log_likelihoods = compute_all_log_likelihoods(z, proto_mu, proto_std, proto_pi)  # [B, C]
    mask = censor_mask(label, censor, proto_mu.shape[0])               # [B, C]

    # Softmax over masked valid classes
    logits = log_likelihoods / temperature                                           # [B, C]
    logits = logits.masked_fill(mask == 0, -1e9)

    # CrossEntropy 只允许正类出现在 mask 中
    loss = F.cross_entropy(logits, label, reduction='mean')
    return loss

def safe_softmax(pi, dim=-1, eps=1e-6):
    """
    数值安全的 softmax 实现，防止 NaN、inf 等问题
    """
    # step1: subtract max for numerical stability
    pi_stable = pi - pi.max(dim=dim, keepdim=True)[0]

    # step2: exp + clamp (防止 inf / underflow)
    exp_pi = torch.exp(pi_stable)
    exp_pi = exp_pi.clamp(min=eps, max=1e4)  # 避免 inf

    # step3: normalize
    sum_exp = exp_pi.sum(dim=dim, keepdim=True).clamp(min=eps)
    softmax_pi = exp_pi / sum_exp

    return softmax_pi


def safe_softmax_with_check(pi, dim=-1, eps=1e-6, name="pi", verbose=True, auto_fix=False):
    """
    更健壮的 softmax 实现，带 NaN 检查和可选修复

    Args:
        pi: tensor
        dim: softmax dim
        eps: small constant
        name: variable name for debug
        verbose: whether to print when NaN found
        auto_fix: if True, replace NaNs with small noise

    Returns:
        softmax_pi: tensor same shape as pi
    """
    if torch.isnan(pi).any():
        if verbose:
            print(f"❌ [safe_softmax] Detected NaN in {name}")
            nan_idx = torch.where(torch.isnan(pi))
            print(f"    → Location of NaN: {nan_idx}")
            print(f"    → Example slice: {pi[nan_idx[0][0]]}")
        if auto_fix:
            with torch.no_grad():
                pi.data[torch.isnan(pi)] = torch.randn_like(pi[torch.isnan(pi)]) * 0.01
            if verbose:
                print(f"✅ [safe_softmax] NaN in {name} has been auto-fixed.")

    # Standard safe-softmax below
    pi_stable = pi - pi.max(dim=dim, keepdim=True)[0]
    exp_pi = torch.exp(pi_stable).clamp(min=eps, max=1e4)
    sum_exp = exp_pi.sum(dim=dim, keepdim=True).clamp(min=eps)
    softmax_pi = exp_pi / sum_exp

    # Check post-softmax
    if torch.isnan(softmax_pi).any() and verbose:
        print(f"❌ [safe_softmax] NaN propagated into final softmax of {name}!")
        raise ValueError("NaN in softmax result. Training may be unstable.")

    return softmax_pi


def get_proto_samples_with_pi(mu, std, pi, sample_times):
    """
    mu:  [C, K, D]
    std: [C, K, D]
    pi:  [C, K]
    sample_times: int
    return: [C, sample_times, D]
    """
    C, K, D = mu.shape
    device = mu.device

    # 1. softmax 归一化 π
    pi = safe_softmax_with_check(pi, dim=-1)  # [C, K]
    # print(pi)
    # 2. 对每个类别采样 sample_times 个 component index，按 pi 分布
    component_ids = []
    for c in range(C):
        dist = torch.distributions.Categorical(pi[c])
        samples = dist.sample((sample_times,))  # [sample_times]
        component_ids.append(samples)
    component_ids = torch.stack(component_ids, dim=0)  # [C, sample_times]

    # 3. 构建采样结果
    # Gather mu and std for each sampled component
    mu_sampled = torch.gather(mu, 1, component_ids.unsqueeze(-1).expand(-1, -1, D))     # [C, sample_times, D]
    std_sampled = torch.gather(std, 1, component_ids.unsqueeze(-1).expand(-1, -1, D))   # [C, sample_times, D]

    # 4. 重参数化采样
    eps = torch.randn((C, sample_times, D), device=device)
    samples = mu_sampled + std_sampled * eps                                            # [C, sample_times, D]
    return samples



def gmm_positive_negative_loss(z, label, censor, proto_mu, proto_std, proto_pi):
    """
    z:       [B, D]
    label:   [B]
    censor:  [B]
    proto_mu, proto_std: [C, K, D]
    proto_pi: [C, K]
    return: scalar loss
    """
    B, D = z.shape
    C, K, _ = proto_mu.shape
    device = z.device

    # Compute all log-likelihoods: [B, C]
    log_likelihoods = compute_all_log_likelihoods(z, proto_mu, proto_std, proto_pi)  # [B, C]

    # Build positive and negative masks (same逻辑 as get_sim_loss)
    class_ids = torch.arange(C, device=device).unsqueeze(0).expand(B, -1)  # [B, C]
    label_expand = label.unsqueeze(1)  # [B, 1]

    # 正类 mask（uncensored: 只 label，censored: label及以后）
    
    pos_mask = torch.where(
        censor.unsqueeze(1) == 0,
        class_ids == label_expand,
        class_ids >= label_expand
    ).float()  # [B, C]

    neg_mask = 1.0 - pos_mask

    # Normalize masks（避免数量不一致）
    pos_count = pos_mask.sum(dim=1, keepdim=True).clamp(min=1)
    neg_count = neg_mask.sum(dim=1, keepdim=True).clamp(min=1)

    pos_score = (log_likelihoods * pos_mask).sum(dim=1) / pos_count.squeeze(1)  # [B]
    neg_score = (log_likelihoods * neg_mask).sum(dim=1) / neg_count.squeeze(1)  # [B]

    # 损失：负的正类得分 + 正的负类得分（目标是拉开两者）
    loss = -pos_score.mean() + neg_score.mean()

    return loss

def sample_from_gmm(mu, std, pi, sample_times, eps=1e-6):
    """
    从 GMM 中采样 prototype
    mu:  [C, K, D]
    std: [C, K, D]
    pi:  [C, K]
    return: [C, sample_times, D]
    """
    C, K, D = mu.shape
    pi = F.softmax(pi, dim=-1)  # [C, K]
    samples_idx = torch.stack([
        torch.multinomial(pi[c], num_samples=sample_times, replacement=True)
        for c in range(C)
    ])  # [C, sample_times]

    # 根据采样索引取 mu/std
    mu_sel = torch.gather(mu, 1, samples_idx.unsqueeze(-1).expand(-1, -1, D))  # [C, sample_times, D]
    std_sel = torch.gather(std, 1, samples_idx.unsqueeze(-1).expand(-1, -1, D))

    eps_noise = torch.randn_like(mu_sel)
    samples = mu_sel + std_sel * eps_noise  # reparameterized sample

    return samples  # [C, sample_times, D]

## 权重 仅作用于方差
def gmm_reparameterize_noise_only(mu: torch.Tensor, std: torch.Tensor, pi_logits: torch.Tensor, temperature=1.0,is_training=True,seed=1):
    """
    只在噪声项使用 Gumbel 重参数化，均值已经固定为 mu: [K, D]

    Args:
        mu: [K, D] — 已通过贝塞尔生成的分量均值
        std: [K, D] — 各分量标准差
        pi_logits: [K] — GMM 分量 logits
        temperature: float — gumbel-softmax 温度

    Returns:
        sample: [K, D] — 每个分量的确定中心 + shared noise
        weighted_sample: [D] — Gumbel weighted 采样点
    """
    K, D = std.shape
    std = F.softplus(std)

    # 处理异常 logits（NaN/Inf）
    if torch.isnan(pi_logits).any() or torch.isinf(pi_logits).any():
        pi_logits = torch.where(torch.isnan(pi_logits), torch.zeros_like(pi_logits), pi_logits)
        pi_logits = torch.clamp(pi_logits, -20, 20)

    # Gumbel-softmax 权重
    if is_training:
        # generator = torch.Generator(device='cuda')  # 选择使用 GPU
        # generator.manual_seed(seed)  # 设置该生成器的种子
        gumbel = F.gumbel_softmax(pi_logits, tau=max(temperature, 1e-3), hard=False)  # [K]
        
    else:
        
        # gumbel = F.gumbel_softmax(pi_logits, tau=max(temperature, 1e-3), hard=False)
        gumbel = F.softmax(pi_logits, dim=-1)
   
    # 重参数化噪声项
    if is_training:
        eps = torch.randn_like(std)  # [K, D]
    else:
        # generator = torch.cuda.manual_seed(seed)  # Set the seed for reproducibility
        eps = torch.normal(torch.zeros_like(std), torch.ones_like(std), generator=torch.cuda.manual_seed(seed)).cuda()
        

    noise = std * eps            # [K, D]
    weighted_noise = torch.sum(gumbel[:, None] * noise, dim=0)  # [D]

    # 输出：共享均值 + Gumbel 噪声扰动
    mu = mu.squeeze(0) if mu.dim() == 2 else mu  # ensure [D]
    sample = mu + weighted_noise  # [D]

    return sample


def gmm_sample_contrastive(z, proto_mu, proto_std, proto_pi, sample_times=4, temperature=0.07):
    """
    GMM + Gumbel-softmax 重参数化对比学习
    z:         [B, D]              — 查询特征
    proto_mu:  [C, K, D]           — 每类 GMM 分量均值
    proto_std: [C, K, D]           — 每类 GMM 分量方差
    proto_pi:  [C, K]              — 每类 GMM 分量权重 logits（未归一化）
    sample_times: int              — 每类采样多少个
    return:    [B, C*sample_times] similarity matrix
    """
    B, D = z.shape
    C, _ = proto_mu.shape

    samples = []
    for c in range(C):
        mu_c = proto_mu[c]         # [K, D]
        # std_c = F.softplus(proto_std[c])  # [K, D]
        std_c =proto_std[c]
        pi_c = proto_pi[c]         # [K]

        samples_c = []
        for _ in range(sample_times):
            # x_c = gmm_reparameterize_safe(mu_c, std_c, pi_c)  # [D]
            ## 仅作用于方差
            x_c=gmm_reparameterize_noise_only(mu_c, std_c, pi_c)  # [D]
            samples_c.append(x_c)

        samples_c = torch.stack(samples_c, dim=0)  # [S, D]
        samples.append(samples_c)

    # [C, S, D] → [C*S, D]
    proto_flat = torch.cat(samples, dim=0)  # [C * S, D]

    # L2 normalize
    proto_norm = F.normalize(proto_flat, dim=-1)  # [C*S, D]
    z_norm = F.normalize(z, dim=-1)               # [B, D]

    # 相似度计算
    sim_matrix = torch.matmul(z_norm, proto_norm.T) / temperature  # [B, C*S]
    
    return sim_matrix  # 对比损失中用作 logits



def gmm_sample_v2(proto_mu, proto_std, proto_pi, sample_times=4, temperature=0.07):
    """
    GMM + Gumbel-softmax 重参数化对比学习
    z:         [B, D]              — 查询特征
    proto_mu:  [C, K, D]           — 每类 GMM 分量均值
    proto_std: [C, K, D]           — 每类 GMM 分量方差
    proto_pi:  [C, K]              — 每类 GMM 分量权重 logits（未归一化）
    sample_times: int              — 每类采样多少个
    return:    [B, C*sample_times] similarity matrix
    """

    C, K,D = proto_mu.shape

    samples = []
    for c in range(C):
        mu_c = proto_mu[c]         # [K, D]
        # std_c = F.softplus(proto_std[c])  # [K, D]
        std_c =proto_std[c]
        pi_c = proto_pi[c]         # [K]

        samples_c = []
        for _ in range(sample_times):
            ##
            x_c = gmm_reparameterize_safe(mu_c, std_c, pi_c)  # [D]
            ## 仅作用于方差
            # x_c=gmm_reparameterize_noise_only(mu_c, std_c, pi_c)  # [D]
            samples_c.append(x_c)

        samples_c = torch.stack(samples_c, dim=0)  # [S, D]
        samples.append(samples_c)

    # [C, S, D] → [C*S, D]
    proto_flat = torch.cat(samples, dim=0)  # [C * S, D]
    
    return proto_flat  # 对比损失中用作 logits


def gmm_sample(proto_mu, proto_std, proto_pi, sample_times=4, temperature=0.07,is_training=True,seed=1):
    """
    GMM + Gumbel-softmax 重参数化对比学习
    z:         [B, D]              — 查询特征
    proto_mu:  [C, K, D]           — 每类 GMM 分量均值
    proto_std: [C, K, D]           — 每类 GMM 分量方差
    proto_pi:  [C, K]              — 每类 GMM 分量权重 logits（未归一化）
    sample_times: int              — 每类采样多少个
    return:    [B, C*sample_times] similarity matrix
    """
    C, _ = proto_mu.shape

    samples = []
    for c in range(C):
        mu_c = proto_mu[c]         # [K, D]
        # std_c = F.softplus(proto_std[c])  # [K, D]
        std_c =proto_std[c]
        pi_c = proto_pi[c]         # [K]

        samples_c = []
        for _ in range(sample_times):
            # x_c = gmm_reparameterize_safe(mu_c, std_c, pi_c)  # [D]
            ## 仅作用于方差
            x_c=gmm_reparameterize_noise_only(mu_c, std_c, pi_c,is_training=is_training,seed=seed)  # [D]
            samples_c.append(x_c)

        samples_c = torch.stack(samples_c, dim=0)  # [S, D]
        samples.append(samples_c)

    # [C, S, D] → [C*S, D]
    proto_flat = torch.cat(samples, dim=0)  # [C * S, D]

    return proto_flat  # 对比损失中用作 logits


def gmm_contrastive_similarity(z, proto_flat, temperature=0.07):
    """
    将查询向量与 GMM 采样得到的 prototype 计算相似度 logits
    z:           [B, D]              查询向量
    proto_flat:  [C * S, D]          每类采样得到的 prototype 向量
    temperature: float               对比学习温度参数
    return:      [B, C * S]          对比 logits
    """
    # L2 normalize
    z_norm = F.normalize(z, dim=-1)               # [B, D]
    proto_norm = F.normalize(proto_flat, dim=-1)  # [C*S, D]

    # 相似度计算
    sim_matrix = torch.matmul(z_norm, proto_norm.T) / temperature  # [B, C*S]
    return sim_matrix


def gmm_reparameterize_safe(mu, std, pi_logits, temperature=1.0):
    """
    Reparameterized sampling from GMM with NaN check
    mu: [K, D]
    std: [K, D]
    pi_logits: [K]
    """
    # [0] 预防 std < 0
    std = F.softplus(std)

    # [1] 检查 logits 是否全是 -inf 或 NaN
    if torch.isnan(pi_logits).any() or torch.isinf(pi_logits).any():
        print("⚠️ Warning: pi_logits contains NaN or inf.")
        print("pi_logits =", pi_logits)
        pi_logits = torch.where(torch.isnan(pi_logits), torch.zeros_like(pi_logits), pi_logits)
        pi_logits = torch.clamp(pi_logits, -20, 20)

    # [2] 防止 temperature 太小
    temperature = max(temperature, 1e-3)

    # [3] Gumbel-softmax sampling
    gumbel = F.gumbel_softmax(pi_logits, tau=temperature, hard=False)  # [K]

    # [4] Gaussian reparameterization
    eps = torch.randn_like(mu)                       # [K, D]
    sample_per_k = mu + std * eps                    # [K, D]

    # [5] Weighted mixture
    x = torch.sum(gumbel.unsqueeze(-1) * sample_per_k, dim=0)  # [D]

    # [6] NaN 检查（可选）
    if torch.isnan(x).any():
        print("🚨 NaN detected in reparameterized GMM sample!")
        print(f"mu: {mu}\nstd: {std}\npi_logits: {pi_logits}\ngumbel: {gumbel}\nsample: {x}")
    
    return x



def gmm_sampled_margin_loss(z, label, censor, proto_mu, proto_std, proto_pi, sample_times=4):
    sim = gmm_sample_contrastive(z, proto_mu, proto_std, proto_pi, sample_times)  # [B, C*S]
    C = proto_pi.shape[0]
    B = z.size(0)

    sim_pos_list = []
    sim_neg_list = []

    for i in range(B):
        cls = label[i].item()
        is_censored = censor[i].item()

        pos_mask = torch.zeros_like(sim[i], dtype=torch.bool)

        if is_censored == 0:
            pos_mask[cls * sample_times : (cls + 1) * sample_times] = True
        else:
            if cls == 0:
                pos_mask[:] = True  # 全部是正类
            else:
                pos_mask[cls * sample_times :] = True

        neg_mask = ~pos_mask

        pos_sim = sim[i][pos_mask]
        neg_sim = sim[i][neg_mask]

        # 避免空集 → 用 0 替代
        sim_pos_list.append(pos_sim.mean() if pos_sim.numel() > 0 else torch.tensor(0.0, device=z.device))
        sim_neg_list.append(neg_sim.mean() if neg_sim.numel() > 0 else torch.tensor(0.0, device=z.device))

    sim_pos = torch.stack(sim_pos_list)  # [B]
    sim_neg = torch.stack(sim_neg_list)  # [B]

    loss = -sim_pos.mean() + sim_neg.mean()
    return loss


def generate_bezier_global_path(mu_endpoints, bezier_ctrl, proto_pi,  n_classes,bezier_curve_fn,):
    """
    使用 GMM 权重先混合出整体均值路径，再进行贝塞尔插值（不对控制点加权）

    Args:
        mu_endpoints: [2, K, D]
        bezier_ctrl:  [n_ctrl, D]
        proto_pi:     [C, K]
        bezier_curve_fn: callable
        n_classes: int

    Returns:
        mu_global_path: [C, D]
    """
    C, K = proto_pi.shape
    D = mu_endpoints.shape[-1]

    pi_0 = torch.softmax(proto_pi[0], dim=-1)  # [K]
    pi_1 = torch.softmax(proto_pi[-1], dim=-1)  # [K]

    # 加权求和（广播乘法 + sum）
    mu0 = torch.sum(pi_0[:, None] * mu_endpoints[0], dim=0)  # [D]
    mu1 = torch.sum(pi_1[:, None] * mu_endpoints[1], dim=0)  # [D]# [K, D]


    # 构造贝塞尔控制序列：[n_ctrl+2, D]
    ctrl_seq = torch.cat([
        mu0.unsqueeze(0),         # 起点
        bezier_ctrl,              # 控制点（未加权）
        mu1.unsqueeze(0)          # 终点
    ], dim=0)

    mu_list = []
    for c in range(n_classes):
        t = c / (n_classes - 1)
        mu = bezier_curve_fn(t, ctrl_seq)  # [D]
        mu_list.append(mu)

    return torch.stack(mu_list, dim=0)  # [C, D]


from math import comb
def bezier_curve(t, control_points):  
    """
    t: scalar float, in [0, 1]
    control_points: Tensor of shape [N, D], where N is the number of control points
    Return: point on Bézier curve at time t, shape [D]
    """
    N = control_points.shape[0] - 1
    terms = [
        comb(N, i) * (1 - t)**(N - i) * t**i * control_points[i]
        for i in range(N + 1)
    ]
    return sum(terms)


def censor_margin_loss(sim, label, censor, sample_times):
    """
    计算 censor-aware margin loss（生存预测的正负相似度设计）

    参数：
    sim          — [B, C*S]，查询特征与各类 sampled GMM prototype 的相似度
    label        — [B]，类别标签（整数）
    censor       — [B]，删失标志，0 表示 uncensored，1 表示 censored
    sample_times — 每类采样 S 个 GMM prototype，用于索引构造正负样本掩码

    返回：
    margin loss — scalar
    """
    B = sim.size(0)
    C_S = sim.size(1)
    C = C_S // sample_times

    sim_pos_list = []
    sim_neg_list = []

    for i in range(B):
        cls = label[i].item()
        is_censored = censor[i].item()

        pos_mask = torch.zeros_like(sim[i], dtype=torch.bool)

        if is_censored == 0:
            # uncensored: 只把自己那一类的 prototypes 当作 positive
            pos_mask[cls * sample_times : (cls + 1) * sample_times] = True
        else:
            # censored: 从当前类别往后的都当作正类（有风险倾向）
            if cls == 0:
                pos_mask[:] = True
            else:
                pos_mask[cls * sample_times :] = True

        neg_mask = ~pos_mask
        pos_sim = sim[i][pos_mask]
        neg_sim = sim[i][neg_mask]

        sim_pos_list.append(pos_sim.mean() if pos_sim.numel() > 0 else torch.tensor(0.0, device=sim.device))
        sim_neg_list.append(neg_sim.mean() if neg_sim.numel() > 0 else torch.tensor(0.0, device=sim.device))

    sim_pos = torch.stack(sim_pos_list)  # [B]
    sim_neg = torch.stack(sim_neg_list)  # [B]
    margin = 0.1  # 可以调节的超参
    loss = F.relu(margin - sim_pos + sim_neg).mean()
    return loss


## 归一化对称
def symmetric_kl_divergence(mu1, std1, mu2, std2, eps=1e-6):
    std1 = std1 + eps; std2 = std2 + eps
    D = mu1.size(1)
    kl1 = torch.sum(torch.log(std2/std1) + (std1**2 + (mu1-mu2)**2)/(2*std2**2) - 0.5, dim=1) / D
    kl2 = torch.sum(torch.log(std1/std2) + (std2**2 + (mu2-mu1)**2)/(2*std1**2) - 0.5, dim=1) / D
    return (kl1 + kl2).mean()


def symmetric_tri_kl(mu1, std1, mu2, std2, mu3, std3, eps=1e-6):
    """
    对称 KL：三者 z_path, z_geno, z_fusion 之间两两对齐的 KL loss
    即：SymKL(z1||z2) + SymKL(z1||z3) + SymKL(z2||z3)
    """
    def sym_kl(mu_a, std_a, mu_b, std_b):
        std_a = std_a + eps
        std_b = std_b + eps
        kl_ab = torch.sum(torch.log(std_b / std_a) + (std_a**2 + (mu_a - mu_b)**2) / (2 * std_b**2) - 0.5, dim=1)
        kl_ba = torch.sum(torch.log(std_a / std_b) + (std_b**2 + (mu_b - mu_a)**2) / (2 * std_a**2) - 0.5, dim=1)
        return (kl_ab + kl_ba) / 2

    # 分别计算三组的对称 KL
    kl_12 = sym_kl(mu1, std1, mu2, std2)  # path ↔ geno
    kl_13 = sym_kl(mu1, std1, mu3, std3)  # path ↔ fusion
    kl_23 = sym_kl(mu2, std2, mu3, std3)  # geno ↔ fusion

    return (kl_12 + kl_13 + kl_23).mean()

def compute_feature_importance(h_path, h_geno):
    # 计算路径和基因的方差，作为它们的“重要性”
    path_importance = torch.var(h_path, dim=0)  # [D]
    geno_importance = torch.var(h_geno, dim=0)  # [D]
    
    # 正则化使其变成权重：越大，权重越高
    total_importance = path_importance + geno_importance
    path_weight = path_importance / total_importance
    geno_weight = geno_importance / total_importance
    
    return path_weight, geno_weight



def reparameterize(mu, std):
    eps = torch.randn_like(std)
    return mu + eps * std



## 单个高斯分布
def get_proto_test(proto_mu, proto_std, sample_times=1,is_training=True,seed=1):
    """
    输入：
      - proto_mu:  [C, 1, D] 或者 [C, D] 形式的类别中心均值
      - proto_std: [C, 1, D] 或者 [C, D] 形式的类别中心标准差
      - sample_times: 采样次数 K

    输出：
      - 如果 sample_times == 1：
          返回 proto = F.normalize(proto_mu.squeeze(1), dim=-1)，形状 [C, D]
      - 否则 (K > 1)：
          返回 shape 为 [K, C, D] 的张量，
          每一行对应一次“重参数化采样后再归一化”的 [C, D]
    """
    # 如果只采一次，直接返回归一化后的 μ
    if sample_times == 1:
        proto = F.normalize(proto_mu.squeeze(1), dim=-1)  # [C, D]
        return proto

    # 否则，对每次采样做重参数化
    proto_list = []
    for _ in range(sample_times):
        # 对 proto_mu、proto_std 都是 [C, 1, D]，这里先 squeeze 出 [C, D]
        mu = proto_mu.squeeze(1)                     # [C, D]
        std = F.softplus(proto_std.squeeze(1) - 5)    # [C, D]，类似你之前用 softplus 做正则
        samp = reparameterize(mu, std)                # [C, D]
        samp_norm = F.normalize(samp, dim=-1)         # [C, D]
        proto_list.append(samp_norm)

    # torch.stack 后得到 [K, C, D]
    return torch.stack(proto_list, dim=0)


