from ast import Lambda
import numpy as np
import pdb
import os
from custom_optims.radam import RAdam


from models.model_BezierSurv_ablation_random_test import BezierSurv


from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv

from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)


#----> pytorch imports
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss
from utils.loss_func import NLLLogistiHazardLoss
import torch.optim as optim



def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually
    
    Args:
        - datasets : tuple
        - cur : Int 
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    if args.val_test:
        train_split, val_split,test_split = datasets
        _save_splits(datasets, ['train', 'val','test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))
        print("Testing on {} samples".format(len(test_split)))
        return train_split,val_split,test_split

    else:
        train_split, val_split = datasets

        _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
        print('Done!')
        print("Training on {} samples".format(len(train_split)))
        print("Validating on {} samples".format(len(val_split)))

        return train_split,val_split



def _init_loss_function(args):
    r"""
    Init the survival loss function
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss
    
    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'NLLLogist':
        loss_fn =NLLLogistiHazardLoss()
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn

def _init_optim(args, model):
    r"""
    Init the optimizer 
    
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
    
    Returns:
        - optimizer : torch optim 
    """
    print('\nInit optimizer ...', end=' ')

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):
    
    print('\nInit Model...', end=' ')
    if args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        omics_input_dim = 4999
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    else:
        omics_input_dim = 0
    
    # omics baselines
    if args.modality == "mlp_per_path":

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "dropout" : args.encoder_dropout, "num_classes" : args.n_classes,
        }
        model = MaskedOmics(**model_dict)

    elif args.modality == "omics":

        model_dict = {
             "input_dim" : omics_input_dim, "projection_dim": 64, "dropout": args.encoder_dropout,
             "n_classes" : args.n_classes,
        }
        model = MLPOmics(**model_dict)

    elif args.modality == "snn":

        model_dict = {
             "omic_input_dim" : omics_input_dim, 
             "n_classes" : args.n_classes,
        }
        model = SNNOmics(**model_dict)

    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion,"n_classes" : args.n_classes,
        }

        model = ABMIL(**model_dict)
    elif args.modality in [ "s4mil_wsi","s4mil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "in_dim":args.encoding_dim,"df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion,"n_classes" : args.n_classes,"dropout":args.encoder_dropout,
        }

        model = S4Model(**model_dict)
    # unimodal and multimodal baselines
    elif args.modality in ["deepmisl_wsi", "deepmisl_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion,"n_classes" : args.n_classes,
        }

        model = DeepMISL(**model_dict)

    elif args.modality in ["dsmil_wsi", "dsmil_wsi_pathways"]:
        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion,"n_classes" : args.n_classes,
        }

        model = DSMIL(**model_dict)

    elif args.modality == "mlp_wsi":
        
        model_dict = {
            "wsi_embedding_dim":args.encoding_dim, "input_dim_omics":omics_input_dim, "dropout":args.encoder_dropout,
            "device": args.device,"num_classes" : args.n_classes,

        }
        model = MLPWSI(**model_dict)

    elif args.modality in ["transmil_wsi", "transmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion,"n_classes" : args.n_classes,
        }

        model = TMIL(**model_dict)

    elif args.modality == "coattn":

        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCATPathways(**model_dict)

    elif args.modality == "coattn_motcat":

        model_dict = {
            'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes,
            "ot_reg":0.1, "ot_tau":0.5, "ot_impl":"pot-uot-l2"
        }
        model = MCATPathwaysMotCat(**model_dict)

    elif args.modality == "coattn_motcat_bezier":

        model_dict = {
            'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes,
            "ot_reg":0.1, "ot_tau":0.5, "ot_impl":"pot-uot-l2"
        }
        model = MCATPathwaysMotCat_bezier(**model_dict)
        
    # survpath 
    elif args.modality == "survpath":

        model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes}

        if args.use_nystrom:
            model = SurvPath_with_nystrom(**model_dict)
        else:
            model = SurvPath(**model_dict)
    elif args.modality == "ProSurv":

        model_dict = {
             "omic_input_dim" : omics_input_dim, "dropout": args.encoder_dropout,
             "mil_model_type": args.mil_model_type, "geno_mlp_type":args.geno_mlp_type,
             "memory_size": args.memory_size,"n_classes" : args.n_classes,
        }
        model = ProSurv(**model_dict)
    elif args.modality == "DisSurv":

        model_dict = {
             "omic_input_dim" : omics_input_dim, "dropout": args.encoder_dropout,
             "mil_model_type": args.mil_model_type, "geno_mlp_type":args.geno_mlp_type,
             "memory_size": args.memory_size
        }
        model = DisSurv(**model_dict)
    elif args.modality == "BezierSurv":

        # model_dict = {
        #      "omic_input_dim" : omics_input_dim, "dropout": args.encoder_dropout,
        #      "mil_model_type": args.mil_model_type, "geno_mlp_type":args.geno_mlp_type,
        #      "memory_size": args.memory_size,'n_classes': args.n_classes
        # }
        model_dict = {
        "omic_input_dim"     : omics_input_dim,
        "dropout"            : args.encoder_dropout,
        "mil_model_type"     : args.mil_model_type,
        "geno_mlp_type"      : args.geno_mlp_type,
        "memory_size"        : args.memory_size,
        "n_classes"          : args.n_classes,
        "fusion"             : args.fusion if hasattr(args, "fusion") else "concat",
        "n_ctrl_points"      : args.n_ctrl_points if hasattr(args, "n_ctrl_points") else 3,

        # === 信息瓶颈模块开关 ===
        "use_ib_path"        : args.use_ib_path,
        "use_ib_geno"        : args.use_ib_geno,
        "use_ib_fusion"      : args.use_ib_fusion,

        # === 协同对齐损失/模态对齐 ===
        "use_kl_align"       : args.use_kl_align,
        "use_align_loss"     : args.use_align_loss,

        # === 对比损失（伪标签正负样本判别）===
        "use_sim_loss"       : args.use_sim_loss,

        # === 是否使用贝塞尔曲线生成 GMM prototype ===
        "use_bezier_gmm"     : args.use_bezier_gmm if hasattr(args, "use_bezier_gmm") else True,
        "seed":args.seed,
        "use_mean_in_testing":args.use_mean_in_testing,
        "prototype_mode" :args.prototype_mode
         }
        model = BezierSurv(**model_dict) 
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    _print_network(args.results_dir, model)

    return model

def _init_loaders(args, train_split, val_split,test_split=None):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """

    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    if args.val_test:
        if test_split:
            test_loader = _get_split_loader(args, test_split,  testing=False, batch_size=1)
        else:
            test_loader = None
    print('Done!')
    if args.val_test:
        return train_loader,val_loader,test_loader
    else:
        return train_loader,val_loader

def _extract_survival_metadata(train_loader, val_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _unpack_data(modality, device, data):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
    
    """
    
    if modality in ["mlp_per_path", "omics", "snn"]:
        # return (torch.zeros((1,1)), omics_tensor, label, event_time, c, clinical_data)
        data_WSI = data[0]
        mask = None
        data_omics = data[1].to(device)
        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    
    elif modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "s4mil_wsi","s4mil_wsi_pathways",
                      "deepmisl_wsi", "deepmisl_wsi_pathways","dsmil_wsi","dsmil_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]

    elif modality in ["coattn", "coattn_motcat","coattn_motcat_bezier"]:
        
        data_WSI = data[0].to(device)
        data_omic1 = data[1].type(torch.FloatTensor).to(device)
        data_omic2 = data[2].type(torch.FloatTensor).to(device)
        data_omic3 = data[3].type(torch.FloatTensor).to(device)
        data_omic4 = data[4].type(torch.FloatTensor).to(device)
        data_omic5 = data[5].type(torch.FloatTensor).to(device)
        data_omic6 = data[6].type(torch.FloatTensor).to(device)
        data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]

        y_disc, event_time, censor, clinical_data_list, mask,patient_id = data[7], data[8], data[9], data[10], data[11],data[12]
        mask = mask.to(device)

    elif modality in ["survpath"]:

        data_WSI = data[0].to(device)

        data_omics = []
        for item in data[1][0]:
            data_omics.append(item.to(device))
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
    elif modality in ["ProSurv","DisSurv","BezierSurv"]:

        data_WSI = data[0].to(device)
        data_omics = data[1].to(device)
        
        if data[6][0,0] == 1:
            mask = None
        else:
            mask = data[6].to(device)

        y_disc, event_time, censor, clinical_data_list ,patient_id= data[2], data[3], data[4], data[5],data[7]

    else:
        raise ValueError('Unsupported modality:', modality)
    
    y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)
    if modality in ["coattn_motcat_bezier","ProSurv","DisSurv","BezierSurv"]:
        return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask,patient_id
    else:
        return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask

def _process_data_and_forward(model, modality, device, data,is_training=True):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple
    
    Returns:
        - out : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
    
    """
    if modality in ["coattn_motcat_bezier","BezierSurv","ProSurv"]:
        data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask,patient_id = _unpack_data(modality, device, data)
    else:
        data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
        
    if modality in ["coattn", "coattn_motcat"]:  
        
        out = model(
            x_path=data_WSI, 
            x_omic1=data_omics[0], 
            x_omic2=data_omics[1], 
            x_omic3=data_omics[2], 
            x_omic4=data_omics[3], 
            x_omic5=data_omics[4], 
            x_omic6=data_omics[5]
            )  
        out = out[0]
    elif modality == 'survpath':

        input_args = {"x_path": data_WSI.to(device)}
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
        input_args["return_attn"] = False
        out = model(**input_args)

    elif modality in ["coattn_motcat_bezier"]:
        out = model(
            x_path=data_WSI, 
            x_omic1=data_omics[0], 
            x_omic2=data_omics[1], 
            x_omic3=data_omics[2], 
            x_omic4=data_omics[3], 
            x_omic5=data_omics[4], 
            x_omic6=data_omics[5],
            label = y_disc,
            censor =censor,
           event_time=event_time,
        #    is_training=is_training
            )  
        # out = out[0]
    elif modality in ["BezierSurv","ProSurv"]:
        input_args = {"x_path": data_WSI.to(device)}
        input_args["x_omic"] = data_omics.to(device)
        input_args["label"] = y_disc
        input_args['censor'] = censor
        input_args['training'] = True
        input_args['input_modality'] = "path_and_geno"
        input_args['return_feature'] = False
        input_args['event_time']=event_time
    
        out = model(**input_args)
    else:
        out = model(
            data_omics = data_omics, 
            data_WSI = data_WSI, 
            mask = mask
            )
        
    if modality in["coattn_motcat_bezier","BezierSurv","ProSurv"]:
        if len(out[0].shape) == 1:
            # out[0] = out[0].unsqueeze(0)
            out = list(out)
            out[0] = out[0].unsqueeze(0)
            out = tuple(out)

    else:
            
        if len(out.shape) == 1:
                out = out.unsqueeze(0)
  
    return out, y_disc, event_time, censor, clinical_data_list


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, 
                   event_time, censor, risk, clinical_data_list):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data

import random
def split_chunk_list(data, batch_size):
    numGroup = data.shape[0] // batch_size + 1
    feat_index = list(range(data.shape[0]))
    random.shuffle(feat_index)
    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
    index_chunk_list = [sst.tolist() for sst in index_chunk_list]
    return index_chunk_list

def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float 
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []

    # one epoch
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()

        
        if modality ==  'coattn_motcat_bezier':
            ## h 为 logits,sim_loss,ib_path_loss,ib_geno_loss
            h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data)
            sim_loss = h[1]
            ib_path_loss = h[2]
            ib_geno_loss = h[3]
            if str(loss_fn)=='NLLSurvLoss()':
                risk_loss = loss_fn(h=h[0], y=y_disc, t=event_time, c=censor) 
            else:
                risk_loss = loss_fn(
                        h[0], y_disc.to(h.device), censor.to(h.device)
                    )
            # 在 epoch 外层设置
            warmup_epochs = 20
            beta_path_max = 1.0   # 控制最终最大压缩程度
            beta_geno_max = 1.0

            eta = min(1.0, epoch / warmup_epochs)  # 从 0 逐渐变为 1
            lambda_path = 1.0 - eta * beta_path_max  # 初始 1.0，逐步收紧信息瓶颈
            lambda_geno = 1.0 - eta * beta_geno_max

            loss = (risk_loss / y_disc.shape[0]) + sim_loss
            loss = loss + lambda_path * ib_path_loss + lambda_geno * ib_geno_loss
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
            
            risk, _ = _calculate_risk(h)

            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

            total_loss += loss_value 
            # if modality==  'coattn_motcat':
            #     loss = loss / 32 + 0
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx % 20) == 0:
                print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
        elif modality ==  'BezierSurv':
            h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data,is_training)
            ## logits, sim_loss, loss_align,ib_path_loss,ib_geno_loss,ib_fusion_loss,kl_align_loss
            sim_loss = h[1]
            loss_align = h[2]
            ib_path_loss = h[3]
            ib_geno_loss = h[4]
            ib_fusion_loss =h[5]
            kl_align_loss =h[6]
            if str(loss_fn)=='NLLSurvLoss()':
                risk_loss = loss_fn(h=h[0], y=y_disc, t=event_time, c=censor) 
            else:
                risk_loss = loss_fn(
                        h[0], y_disc.to(h.device), censor.to(h.device)
                    )
            # 在 epoch 外层设置
            
            ib_path_weight = (ib_path_loss.detach() < ib_path_thresh).float()  # shape [B]
            ib_geno_weight = (ib_geno_loss.detach() < ib_geno_thresh).float()
            ib_fusion_weight = (ib_fusion_loss.detach() < ib_fusion_thresh).float()
            # if ib_path_weight.sum()>1:
            #     print("确实用")
            # 分别加权 loss
            weighted_ib_path_loss = (ib_path_weight * ib_path_loss).mean()
            weighted_ib_geno_loss = (ib_geno_weight * ib_geno_loss).mean()
            weighted_ib_fusion_loss = (ib_fusion_weight * ib_fusion_loss).mean()
            
            weighted_risk_loss = risk_loss / y_disc.shape[0]

         

            lambda_path = eta * beta_path_max
            lambda_geno = eta * beta_geno_max
            lambda_fusion = eta * beta_fusion_max

            
            λ_ib = 0.5          # weight for IB losses
            λ_kl = 5e-5          # weight for KL symmetric alignment
            λ_align = 0.2       # weight for latent alignment
            λ_sim = 1.0         # weight for GMM contrastive

            # === Step 5: Total loss ===
            loss = (
                weighted_risk_loss
                + λ_ib * (lambda_path * weighted_ib_path_loss + lambda_geno * weighted_ib_geno_loss + lambda_fusion * weighted_ib_fusion_loss)
                + λ_kl * kl_align_loss
                + λ_align * loss_align
                + λ_sim * sim_loss
            )

            # loss = weighted_risk_loss + lambda_path * weighted_ib_path_loss + lambda_geno * weighted_ib_geno_loss + lambda_fusion*weighted_ib_fusion_loss+sim_loss+loss_align

            # loss = (risk_loss / y_disc.shape[0]) + sim_loss
            # loss = loss + lambda_path * ib_path_loss + lambda_geno * ib_geno_loss
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
            
            risk, _ = _calculate_risk(h[0])

            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

            total_loss += loss_value 
            # if modality==  'coattn_motcat':
            #     loss = loss / 32 + 0
            loss.backward()
            optimizer.step()
            scheduler.step()

            # if (batch_idx % 20) == 0:
            #     print(f"[Batch {batch_idx}] Loss={loss_value:.4f} | Risk={weighted_risk_loss.item():.4f} | IB=({weighted_ib_path_loss.item():.4f}, {weighted_ib_geno_loss.item():.4f}, {weighted_ib_fusion_loss.item():.4f}) | KL={kl_align_loss.item():.4f} | Align={loss_align.item():.4f} | Sim={sim_loss.item():.4f}")

            if (batch_idx % 20) == 0:
                weighted_ib_total = (
                    lambda_path * weighted_ib_path_loss.item() +
                    lambda_geno * weighted_ib_geno_loss.item() +
                    lambda_fusion * weighted_ib_fusion_loss.item()
                )
                print(f"[Batch {batch_idx}] "
                      f"Loss={loss_value:.4f} | "
                      f"Risk={weighted_risk_loss.item():.4f} | "
                      f"IB(λ*): {λ_ib * weighted_ib_total:.4f} "
                      f"[Path={λ_ib * lambda_path * weighted_ib_path_loss.item():.4f}, "
                      f"Geno={λ_ib * lambda_geno * weighted_ib_geno_loss.item():.4f}, "
                      f"Fusion={λ_ib * lambda_fusion * weighted_ib_fusion_loss.item():.4f}] | "
                      f"KL(λ={λ_kl})={λ_kl * kl_align_loss.item():.4f} | "
                      f"Align(λ={λ_align})={λ_align * loss_align.item():.4f} | "
                      f"Sim(λ={λ_sim})={λ_sim * sim_loss.item():.4f}")


        else:  
            h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, 
                                                                                          modality, device, 
                                                                                          data)
            # print(h)
            if modality in ["ProSurv"]:
                h, sim_loss, loss_align = h
                other_loss = 0.2*loss_align +0.2 *sim_loss 
            if str(loss_fn)=='NLLSurvLoss()':
                loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor) 
            else:
                loss = loss_fn(
                        h, y_disc.to(h.device), censor.to(h.device)
                    )
            if modality in ["ProSurv"]:
                loss =loss+other_loss
            # weight2 = args.sim_loss
            # weight3 = args.align_loss
            # if args.use_align_loss:
            #     loss = surv_loss + weight2*sim_loss + weight3*loss_align
            # else:
            #     loss = surv_loss + weight2*sim_loss

            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
            
            risk, _ = _calculate_risk(h)

            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

            total_loss += loss_value 
            # if modality==  'coattn_motcat':
            #     loss = loss / 32 + 0
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx % 20) == 0:
                print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    return c_index, total_loss

def update_difficulty_buffer(model, loader, modality, loss_fn, difficulty_buffer):
    model.eval()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        if modality in ["coattn_motcat_bezier"]:
            for batch_idx, data in enumerate(loader):
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask, patient_ids = _unpack_data(modality, device, data)
                for i, pid in enumerate(patient_ids):
                    single_data = {
                        "x_path": data_WSI[i],
                        "x_omic1": data_omics[0][i],
                        "x_omic2": data_omics[1][i],
                        "x_omic3": data_omics[2][i],
                        "x_omic4": data_omics[3][i],
                        "x_omic5": data_omics[4][i],
                        "x_omic6": data_omics[5][i],
                        "label": y_disc[i].unsqueeze(0),
                        "censor": censor[i].unsqueeze(0),
                        "event_time": event_time[i].unsqueeze(0)
                    }
                    logits, sim_loss, ib_path_loss, ib_geno_loss = model(**single_data)
                    if pid not in difficulty_buffer:
                        difficulty_buffer[pid] = {}
                        difficulty_buffer[pid]['ib_path'] = ib_path_loss[0].item()
                        difficulty_buffer[pid]['ib_geno'] = ib_geno_loss[0].item()
                    else:
                        alpha = 0.9
                        difficulty_buffer[pid]['ib_path'] = alpha * difficulty_buffer[pid].get('ib_path', 0.0) + (1 - alpha) * ib_path_loss[0].item()
                        difficulty_buffer[pid]['ib_geno'] = alpha * difficulty_buffer[pid].get('ib_geno', 0.0) + (1 - alpha) * ib_geno_loss[0].item()
        elif modality in ["BezierSurv"]:
            for batch_idx, data in enumerate(loader):
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask, patient_ids = _unpack_data(modality, device, data)
                for i, pid in enumerate(patient_ids):
                    single_data = {
                        "x_path": data_WSI[i].unsqueeze(0),
                        "x_omic": data_omics[i],
                        "input_modality" : "path_and_geno",
                        "label": y_disc[i].unsqueeze(0),
                        "censor": censor[i].unsqueeze(0),
                        "event_time": event_time[i].unsqueeze(0),
                        "training": False,
                        "return_feature" : False
                    }
                    ## 注意顺序，fusion 在最后
                    logits, ib_path_loss, ib_geno_loss,ib_fusion_loss= model(**single_data)
                    if pid not in difficulty_buffer:
                        difficulty_buffer[pid] = {}
                        difficulty_buffer[pid]['ib_path'] = ib_path_loss[0].item()
                        difficulty_buffer[pid]['ib_geno'] = ib_geno_loss[0].item()
                        difficulty_buffer[pid]['ib_fusion'] = ib_fusion_loss[0].item()
                        
                    else:
                        alpha = 0.9
                        difficulty_buffer[pid]['ib_path'] = alpha * difficulty_buffer[pid].get('ib_path', 0.0) + (1 - alpha) * ib_path_loss[0].item()
                        difficulty_buffer[pid]['ib_geno'] = alpha * difficulty_buffer[pid].get('ib_geno', 0.0) + (1 - alpha) * ib_geno_loss[0].item()
                        difficulty_buffer[pid]['ib_fusion'] = alpha * difficulty_buffer[pid].get('ib_fusion', 0.0) + (1 - alpha) * ib_fusion_loss[0].item()
                        
    model.train()
    return difficulty_buffer

def _train_loop_survival_self_paced(epoch, model, modality, loader, optimizer, 
                                    scheduler, loss_fn,difficulty_buffer,is_training=True,use_self_paced_ablation=False,
                                     warmup_epochs = 30,spl_on_censored_only = True,use_modal_balance=False):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float 
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    
    # with torch.no_grad():
    #     for batch_idx, data in enumerate(loader):
    #         if modality ==  'coattn_motcat_bezier':
    #             data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask,patient_ids = _unpack_data(modality, device, data)
    #             logits,sim_loss,ib_path_loss,ib_geno_loss = model(
    #                 x_path=data_WSI, 
    #                 x_omic1=data_omics[0], 
    #                 x_omic2=data_omics[1], 
    #                 x_omic3=data_omics[2], 
    #                 x_omic4=data_omics[3], 
    #                 x_omic5=data_omics[4], 
    #                 x_omic6=data_omics[5],
    #                 label = y_disc,
    #                 censor =censor,
    #                 event_time=event_time
    #             )  
    #             # risk_loss =[]
    #             # for i in range(logits.shape[0]):
    #             #     risk_loss.append(loss_fn(h=logits[i], y=y_disc[i], t=event_time[i], c=censor[i]))
    #             for i, pid in enumerate(patient_ids):  # sample_ids 是长度为 B 的字符串列表
    #                 if pid not in difficulty_buffer:
    #                     difficulty_buffer[pid] = {}
                    
    #                 alpha = 0.9  # EMA 参数（历史平滑）

    #                 # 更新不同 loss 分量（risk, ib_path, ib_geno）
    #                 # difficulty_buffer[pid]['risk'] = alpha * difficulty_buffer[pid].get('risk', 0.0) + (1 - alpha) * risk_loss[i].item()
    #                 difficulty_buffer[pid]['ib_path'] = alpha * difficulty_buffer[pid].get('ib_path', 0.0) + (1 - alpha) * ib_path_loss[i].item()
    #                 difficulty_buffer[pid]['ib_geno'] = alpha * difficulty_buffer[pid].get('ib_geno', 0.0) + (1 - alpha) * ib_geno_loss[i].item()
    

    beta_path_max = 1.0   # 控制最终最大压缩程度
    beta_geno_max = 1.0
    beta_fusion_max = 1.0
    # warmup_epochs = 30 ## 可以手动设置，但是好像 这个创新点 似乎改变不了太多
    if spl_on_censored_only:
        eta = min(1.0, (epoch+2)/ warmup_epochs)
    else:
        eta = min(1.0, (epoch+2)/ warmup_epochs)  # 最多只采用百分之八十的样本
    
    # 获取每个样本的IB难度分位阈值
    if modality in ["BezierSurv"]: ## 目前只有BezierSurv支持 
        if len(difficulty_buffer) > 0 and not(use_self_paced_ablation):  ## 消融实验 默认开启 这个地方
            ib_path_all = torch.tensor([v['ib_path'] for v in difficulty_buffer.values()])
            ib_geno_all = torch.tensor([v['ib_geno'] for v in difficulty_buffer.values()])
            ib_fusion_all = torch.tensor([v['ib_fusion'] for v in difficulty_buffer.values()])
            ib_path_thresh = torch.quantile(ib_path_all, eta)
            ib_geno_thresh = torch.quantile(ib_geno_all, eta)
            ib_fusion_thresh = torch.quantile(ib_fusion_all,eta)
        else:
            ib_path_thresh = ib_geno_thresh =ib_fusion_thresh= torch.tensor(float('inf'))
    else:
        if len(difficulty_buffer) > 0:
            ib_path_all = torch.tensor([v['ib_path'] for v in difficulty_buffer.values()])
            ib_geno_all = torch.tensor([v['ib_geno'] for v in difficulty_buffer.values()])
            ib_path_thresh = torch.quantile(ib_path_all, eta)
            ib_geno_thresh = torch.quantile(ib_geno_all, eta)
        else:
            ib_path_thresh = ib_geno_thresh = torch.tensor(float('inf'))

    # one epoch
    for batch_idx, data in enumerate(loader):
        
        optimizer.zero_grad()
        if modality ==  'coattn_motcat_bezier':
            ## h 为 logits,sim_loss,ib_path_loss,ib_geno_loss
            h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data,is_training)
            sim_loss = h[1]
            ib_path_loss = h[2]
            ib_geno_loss = h[3]
            if str(loss_fn)=='NLLSurvLoss()':
                risk_loss = loss_fn(h=h[0], y=y_disc, t=event_time, c=censor) 
            else:
                risk_loss = loss_fn(
                        h[0], y_disc.to(h.device), censor.to(h.device)
                    )
            # 在 epoch 外层设置
            
            ib_path_weight = (ib_path_loss.detach() < ib_path_thresh).float()  # shape [B]
            ib_geno_weight = (ib_geno_loss.detach() < ib_geno_thresh).float()
            
            
            # if ib_path_weight.sum()>1:
            #     print("确实用")
            # 分别加权 loss
            weighted_ib_path_loss = (ib_path_weight * ib_path_loss).mean()
            weighted_ib_geno_loss = (ib_geno_weight * ib_geno_loss).mean()
            weighted_risk_loss = risk_loss / y_disc.shape[0]

         
            lambda_path = 1.0 - eta * beta_path_max
            lambda_geno = 1.0 - eta * beta_geno_max

            loss = weighted_risk_loss + lambda_path * weighted_ib_path_loss + lambda_geno * weighted_ib_geno_loss + sim_loss

            # loss = (risk_loss / y_disc.shape[0]) + sim_loss
            # loss = loss + lambda_path * ib_path_loss + lambda_geno * ib_geno_loss
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
            
            risk, _ = _calculate_risk(h[0])

            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

            total_loss += loss_value 
            # if modality==  'coattn_motcat':
            #     loss = loss / 32 + 0
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx % 20) == 0:
                print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))

        elif modality ==  'BezierSurv':
            h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data,is_training)
            ## logits, sim_loss, loss_align,ib_path_loss,ib_geno_loss,ib_fusion_loss,kl_align_loss
            censor_mask = censor.float().detach()  # shape [B]
            sim_loss = h[1]
            loss_align = h[2]
            ib_path_loss = h[3]
            ib_geno_loss = h[4]
            ib_fusion_loss =h[5]
            kl_align_loss =h[6]
            if str(loss_fn)=='NLLSurvLoss()':
                risk_loss = loss_fn(h=h[0], y=y_disc, t=event_time, c=censor) 
            else:
                risk_loss = loss_fn(
                        h[0], y_disc.to(h.device), censor.to(h.device)
                    )
            # 在 epoch 外层设置
            
            # ib_path_weight = (ib_path_loss.detach() < ib_path_thresh).float()  # shape [B]
            # ib_geno_weight = (ib_geno_loss.detach() < ib_geno_thresh).float()
            # ib_fusion_weight = (ib_fusion_loss.detach() < ib_fusion_thresh).float()

           

            # === Step 2: 基于“可信度”计算原始分数（只对 mask=1 的样本有效，mask=0 的赋 0）===
            #       可信度 = 1 / (ib_loss + eps)，然后乘以 mask
            # === Step 1: 原 SPL 二值掩码 ===
            if use_modal_balance:
                if spl_on_censored_only:
                    ib_path_mask = ((ib_path_loss.detach() < ib_path_thresh).float() * censor_mask) + (1.0 - censor_mask)
                    ib_geno_mask = ((ib_geno_loss.detach() < ib_geno_thresh).float() * censor_mask) + (1.0 - censor_mask)
                    ib_fusion_mask = ((ib_fusion_loss.detach() < ib_fusion_thresh).float() * censor_mask) + (1.0 - censor_mask)
                else:
                    ib_path_mask = (ib_path_loss.detach() < ib_path_thresh).float()
                    ib_geno_mask = (ib_geno_loss.detach() < ib_geno_thresh).float()
                    ib_fusion_mask = (ib_fusion_loss.detach() < ib_fusion_thresh).float()
                eps = 1e-6
                raw_p = ib_path_mask * (1.0 / (ib_path_loss.detach() + eps))  # [B]
                raw_g = ib_geno_mask * (1.0 / (ib_geno_loss.detach() + eps))  # [B]
                # 如果两路都被 mask 掉（都为0），那么 raw_p=raw_g=0
                # 我们在归一化时加 eps 避免除 0 错误

                # === Step 3: 归一化到两路权重之和 = 1（只对 path/gen 路）===
                sum_pg = raw_p + raw_g + eps  # [B]
                alpha_p = raw_p / sum_pg       # [B]
                alpha_g = raw_g / sum_pg       # [B]
                # 如果 sum_pg=eps（即 raw_p=raw_g=0），那么 alpha_p=alpha_g≈0
                # 这时我们希望至少让它们均分一点权重，可设最小下限 w_min：
                w_min = 0.20  # 最低给每路至少 5% 权重
                alpha_p = torch.clamp(alpha_p, min=w_min, max=1.0-w_min)
                alpha_g = torch.clamp(alpha_g, min=w_min, max=1.0-w_min)
                # 重新归一化
                sum2 = alpha_p + alpha_g
                alpha_p = alpha_p / sum2
                alpha_g = alpha_g / sum2
                # 这样即使 raw_p=raw_g=0，clamp 后 alpha_p=alpha_g=0.5

                # === Step 4: 用 α_p, α_g 重新加权 path/gen IB loss ===
                # 注意用 unsqueeze(-1) 把 [B] 变为 [B,1]，再广播到特征的形状通常是 [B] 直接相乘也行
                weighted_ib_path_loss = (alpha_p * ib_path_loss).mean()
                weighted_ib_geno_loss = (alpha_g * ib_geno_loss).mean()
                # 对 fusion IB 单独保留原来的 SPL 二值掩码方式
                weighted_ib_fusion_loss = (ib_fusion_mask * ib_fusion_loss).mean()
                # === Step 5: 风险损失保持不变 ===
                weighted_risk_loss = risk_loss / y_disc.shape[0]
            else:
             # === 原来没有“模态平衡”时的 SPL 逻辑（完全不变） ===
            # if ib_path_weight.sum()>1:
            #     print("确实用")
            # 分别加权 loss
            ## 原始版本
             # 仅删失样本使用 SPL 策略（原始版本）
                if spl_on_censored_only:
                    ib_path_weight = ((ib_path_loss.detach() < ib_path_thresh).float() * censor_mask) + (1.0 - censor_mask)
                    ib_geno_weight = ((ib_geno_loss.detach() < ib_geno_thresh).float() * censor_mask) + (1.0 - censor_mask)
                    ib_fusion_weight = ((ib_fusion_loss.detach() < ib_fusion_thresh).float() * censor_mask) + (1.0 - censor_mask)
                else:
                    ib_path_weight = (ib_path_loss.detach() < ib_path_thresh).float()
                    ib_geno_weight = (ib_geno_loss.detach() < ib_geno_thresh).float()
                    ib_fusion_weight = (ib_fusion_loss.detach() < ib_fusion_thresh).float()
                weighted_ib_path_loss = (ib_path_weight * ib_path_loss).mean()
                weighted_ib_geno_loss = (ib_geno_weight * ib_geno_loss).mean()
                weighted_ib_fusion_loss = (ib_fusion_weight * ib_fusion_loss).mean()
                weighted_risk_loss = risk_loss / y_disc.shape[0]

            lambda_path = 1
            lambda_geno = 1
            lambda_fusion = 1

            λ_ib = 0.5          # weight for IB losses
            λ_kl = 5e-5          # weight for KL symmetric alignment
            λ_align = 0.2       # weight for latent alignment
            λ_sim = 10         # weight for GMM contrastive

            loss = (
                weighted_risk_loss
                + λ_ib * (lambda_path * weighted_ib_path_loss + lambda_geno * weighted_ib_geno_loss + lambda_fusion * weighted_ib_fusion_loss)
                + λ_kl * kl_align_loss
                + λ_align * loss_align
                + λ_sim * sim_loss
            )

            # loss = weighted_risk_loss + lambda_path * weighted_ib_path_loss + lambda_geno * weighted_ib_geno_loss + lambda_fusion*weighted_ib_fusion_loss+sim_loss+loss_align

            # loss = (risk_loss / y_disc.shape[0]) + sim_loss
            # loss = loss + lambda_path * ib_path_loss + lambda_geno * ib_geno_loss
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
            
            risk, _ = _calculate_risk(h[0])

            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

            total_loss += loss_value 
            # if modality==  'coattn_motcat':
            #     loss = loss / 32 + 0
            loss.backward()
            optimizer.step()
            scheduler.step()

            # if (batch_idx % 20) == 0:
            #     print(f"[Batch {batch_idx}] Loss={loss_value:.4f} | Risk={weighted_risk_loss.item():.4f} | IB=({weighted_ib_path_loss.item():.4f}, {weighted_ib_geno_loss.item():.4f}, {weighted_ib_fusion_loss.item():.4f}) | KL={kl_align_loss.item():.4f} | Align={loss_align.item():.4f} | Sim={sim_loss.item():.4f}")

            if (batch_idx % 20) == 0:
                weighted_ib_total = (
                    lambda_path * weighted_ib_path_loss.item() +
                    lambda_geno * weighted_ib_geno_loss.item() +
                    lambda_fusion * weighted_ib_fusion_loss.item()
                )
                print(f"[Batch {batch_idx}] "
                      f"Loss={loss_value:.4f} | "
                      f"Risk={weighted_risk_loss.item():.4f} | "
                      f"IB(λ*): {λ_ib * weighted_ib_total:.4f} "
                      f"[Path={λ_ib * lambda_path * weighted_ib_path_loss.item():.4f}, "
                      f"Geno={λ_ib * lambda_geno * weighted_ib_geno_loss.item():.4f}, "
                      f"Fusion={λ_ib * lambda_fusion * weighted_ib_fusion_loss.item():.4f}] | "
                      f"KL(λ={λ_kl})={λ_kl * kl_align_loss.item():.4f} | "
                      f"Align(λ={λ_align})={λ_align * loss_align.item():.4f} | "
                      f"Sim(λ={λ_sim})={λ_sim * sim_loss.item():.4f}")
                
                


        else:  
            h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data)
            if str(loss_fn)=='NLLSurvLoss()':
                loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor) 
            else:
                loss = loss_fn(
                        h, y_disc.to(h.device), censor.to(h.device)
                    )
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
            
            risk, _ = _calculate_risk(h)

            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

            total_loss += loss_value 
            # if modality==  'coattn_motcat':
            #     loss = loss / 32 + 0
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx % 20) == 0:
                print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    return c_index, total_loss

## wandb测试版本
def wandb_train_loop_survival_self_paced(epoch, model, modality, loader, optimizer, 
                                    scheduler, loss_fn,difficulty_buffer,is_training=True,use_self_paced_ablation=False,
                                     warmup_epochs = 30,spl_on_censored_only = True,use_modal_balance=False,
                                     wandb_run=None,use_wandb=False,use_tensorboard=False):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float 
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    total_loss = 0.
    
    all_risk_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    

    beta_path_max = 1.0   # 控制最终最大压缩程度
    beta_geno_max = 1.0
    beta_fusion_max = 1.0
    # warmup_epochs = 30 ## 可以手动设置，但是好像 这个创新点 似乎改变不了太多
    if spl_on_censored_only:
        eta = min(1.0, (epoch+2)/ warmup_epochs)
    else:
        eta = min(1.0, (epoch+2)/ warmup_epochs)  # 最多只采用百分之八十的样本
    
    # 获取每个样本的IB难度分位阈值
    if modality in ["BezierSurv"]: ## 目前只有BezierSurv支持 
        if len(difficulty_buffer) > 0 and not(use_self_paced_ablation):  ## 消融实验 默认开启 这个地方
            ib_path_all = torch.tensor([v['ib_path'] for v in difficulty_buffer.values()])
            ib_geno_all = torch.tensor([v['ib_geno'] for v in difficulty_buffer.values()])
            ib_fusion_all = torch.tensor([v['ib_fusion'] for v in difficulty_buffer.values()])
            ib_path_thresh = torch.quantile(ib_path_all, eta)
            ib_geno_thresh = torch.quantile(ib_geno_all, eta)
            ib_fusion_thresh = torch.quantile(ib_fusion_all,eta)
        else:
            ib_path_thresh = ib_geno_thresh =ib_fusion_thresh= torch.tensor(float('inf'))
    else:
        if len(difficulty_buffer) > 0:
            ib_path_all = torch.tensor([v['ib_path'] for v in difficulty_buffer.values()])
            ib_geno_all = torch.tensor([v['ib_geno'] for v in difficulty_buffer.values()])
            ib_path_thresh = torch.quantile(ib_path_all, eta)
            ib_geno_thresh = torch.quantile(ib_geno_all, eta)
        else:
            ib_path_thresh = ib_geno_thresh = torch.tensor(float('inf'))



    log_every = int(len(loader)*0.1)
    # 用于“每 log_every 个样本一窗”的累加器
    window_loss_sum        = 0.0
    window_risk_sum        = 0.0
    window_sim_sum         = 0.0
    window_align_sum       = 0.0
    window_kl_sum          = 0.0
    window_ib_path_sum     = 0.0
    window_ib_geno_sum     = 0.0
    window_ib_fusion_sum   = 0.0
    window_ib_path_active  = 0.0
    window_ib_geno_active  = 0.0
    window_ib_fusion_active= 0.0
    window_count           = 0
    num_batches = len(loader)
    # one epoch
    for batch_idx, data in enumerate(loader):
        global_step = epoch * num_batches + batch_idx   # 全局 step 计数

        optimizer.zero_grad()
        if modality ==  'coattn_motcat_bezier':
            ## h 为 logits,sim_loss,ib_path_loss,ib_geno_loss
            h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data,is_training)
            sim_loss = h[1]
            ib_path_loss = h[2]
            ib_geno_loss = h[3]
            if str(loss_fn)=='NLLSurvLoss()':
                risk_loss = loss_fn(h=h[0], y=y_disc, t=event_time, c=censor) 
            else:
                risk_loss = loss_fn(
                        h[0], y_disc.to(h.device), censor.to(h.device)
                    )
            # 在 epoch 外层设置
            
            ib_path_weight = (ib_path_loss.detach() < ib_path_thresh).float()  # shape [B]
            ib_geno_weight = (ib_geno_loss.detach() < ib_geno_thresh).float()
            
            
            # if ib_path_weight.sum()>1:
            #     print("确实用")
            # 分别加权 loss
            weighted_ib_path_loss = (ib_path_weight * ib_path_loss).mean()
            weighted_ib_geno_loss = (ib_geno_weight * ib_geno_loss).mean()
            weighted_risk_loss = risk_loss / y_disc.shape[0]

         
            lambda_path = 1.0 - eta * beta_path_max
            lambda_geno = 1.0 - eta * beta_geno_max

            loss = weighted_risk_loss + lambda_path * weighted_ib_path_loss + lambda_geno * weighted_ib_geno_loss + sim_loss

            # loss = (risk_loss / y_disc.shape[0]) + sim_loss
            # loss = loss + lambda_path * ib_path_loss + lambda_geno * ib_geno_loss
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
            
            risk, _ = _calculate_risk(h[0])

            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

            total_loss += loss_value 
            # if modality==  'coattn_motcat':
            #     loss = loss / 32 + 0
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx % 20) == 0:
                print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))

        elif modality ==  'BezierSurv':
            h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data,is_training)
            ## logits, sim_loss, loss_align,ib_path_loss,ib_geno_loss,ib_fusion_loss,kl_align_loss
            censor_mask = censor.float().detach()  # shape [B]
            sim_loss = h[1]
            loss_align = h[2]
            ib_path_loss = h[3]
            ib_geno_loss = h[4]
            ib_fusion_loss =h[5]
            kl_align_loss =h[6]
            if str(loss_fn)=='NLLSurvLoss()':
                risk_loss = loss_fn(h=h[0], y=y_disc, t=event_time, c=censor) 
            else:
                risk_loss = loss_fn(
                        h[0], y_disc.to(h.device), censor.to(h.device)
                    )
            # 在 epoch 外层设置
            
            # ib_path_weight = (ib_path_loss.detach() < ib_path_thresh).float()  # shape [B]
            # ib_geno_weight = (ib_geno_loss.detach() < ib_geno_thresh).float()
            # ib_fusion_weight = (ib_fusion_loss.detach() < ib_fusion_thresh).float()

           

            # === Step 2: 基于“可信度”计算原始分数（只对 mask=1 的样本有效，mask=0 的赋 0）===
            #       可信度 = 1 / (ib_loss + eps)，然后乘以 mask
            # === Step 1: 原 SPL 二值掩码 ===
            if use_modal_balance:
                if spl_on_censored_only:
                    ib_path_mask = ((ib_path_loss.detach() < ib_path_thresh).float() * censor_mask) + (1.0 - censor_mask)
                    ib_geno_mask = ((ib_geno_loss.detach() < ib_geno_thresh).float() * censor_mask) + (1.0 - censor_mask)
                    ib_fusion_mask = ((ib_fusion_loss.detach() < ib_fusion_thresh).float() * censor_mask) + (1.0 - censor_mask)
                else:
                    ib_path_mask = (ib_path_loss.detach() < ib_path_thresh).float()
                    ib_geno_mask = (ib_geno_loss.detach() < ib_geno_thresh).float()
                    ib_fusion_mask = (ib_fusion_loss.detach() < ib_fusion_thresh).float()
                eps = 1e-6
                raw_p = ib_path_mask * (1.0 / (ib_path_loss.detach() + eps))  # [B]
                raw_g = ib_geno_mask * (1.0 / (ib_geno_loss.detach() + eps))  # [B]
                # 如果两路都被 mask 掉（都为0），那么 raw_p=raw_g=0
                # 我们在归一化时加 eps 避免除 0 错误

                # === Step 3: 归一化到两路权重之和 = 1（只对 path/gen 路）===
                sum_pg = raw_p + raw_g + eps  # [B]
                alpha_p = raw_p / sum_pg       # [B]
                alpha_g = raw_g / sum_pg       # [B]
                # 如果 sum_pg=eps（即 raw_p=raw_g=0），那么 alpha_p=alpha_g≈0
                # 这时我们希望至少让它们均分一点权重，可设最小下限 w_min：
                w_min = 0.20  # 最低给每路至少 5% 权重
                alpha_p = torch.clamp(alpha_p, min=w_min, max=1.0-w_min)
                alpha_g = torch.clamp(alpha_g, min=w_min, max=1.0-w_min)
                # 重新归一化
                sum2 = alpha_p + alpha_g
                alpha_p = alpha_p / sum2
                alpha_g = alpha_g / sum2
                # 这样即使 raw_p=raw_g=0，clamp 后 alpha_p=alpha_g=0.5

                # === Step 4: 用 α_p, α_g 重新加权 path/gen IB loss ===
                # 注意用 unsqueeze(-1) 把 [B] 变为 [B,1]，再广播到特征的形状通常是 [B] 直接相乘也行
                weighted_ib_path_loss = (alpha_p * ib_path_loss).mean()
                weighted_ib_geno_loss = (alpha_g * ib_geno_loss).mean()
                # 对 fusion IB 单独保留原来的 SPL 二值掩码方式
                weighted_ib_fusion_loss = (ib_fusion_mask * ib_fusion_loss).mean()
                # === Step 5: 风险损失保持不变 ===
                weighted_risk_loss = risk_loss / y_disc.shape[0]
            else:
             # === 原来没有“模态平衡”时的 SPL 逻辑（完全不变） ===
            # if ib_path_weight.sum()>1:
            #     print("确实用")
            # 分别加权 loss
            ## 原始版本
             # 仅删失样本使用 SPL 策略（原始版本）
                if spl_on_censored_only:
                    ib_path_weight = ((ib_path_loss.detach() < ib_path_thresh).float() * censor_mask) + (1.0 - censor_mask)
                    ib_geno_weight = ((ib_geno_loss.detach() < ib_geno_thresh).float() * censor_mask) + (1.0 - censor_mask)
                    ib_fusion_weight = ((ib_fusion_loss.detach() < ib_fusion_thresh).float() * censor_mask) + (1.0 - censor_mask)
                else:
                    ib_path_weight = (ib_path_loss.detach() < ib_path_thresh).float()
                    ib_geno_weight = (ib_geno_loss.detach() < ib_geno_thresh).float()
                    ib_fusion_weight = (ib_fusion_loss.detach() < ib_fusion_thresh).float()
                weighted_ib_path_loss = (ib_path_weight * ib_path_loss).mean()
                weighted_ib_geno_loss = (ib_geno_weight * ib_geno_loss).mean()
                weighted_ib_fusion_loss = (ib_fusion_weight * ib_fusion_loss).mean()
                weighted_risk_loss = risk_loss / y_disc.shape[0]

            lambda_path = 1
            lambda_geno = 1
            lambda_fusion = 1

            λ_ib = 0.5          # weight for IB losses
            λ_kl = 5e-5          # weight for KL symmetric alignment
            λ_align = 0.2       # weight for latent alignment
            λ_sim = 10         # weight for GMM contrastive

            loss = (
                weighted_risk_loss
                + λ_ib * (lambda_path * weighted_ib_path_loss + lambda_geno * weighted_ib_geno_loss + lambda_fusion * weighted_ib_fusion_loss)
                + λ_kl * kl_align_loss
                + λ_align * loss_align
                + λ_sim * sim_loss
            )

            # loss = weighted_risk_loss + lambda_path * weighted_ib_path_loss + lambda_geno * weighted_ib_geno_loss + lambda_fusion*weighted_ib_fusion_loss+sim_loss+loss_align

            # loss = (risk_loss / y_disc.shape[0]) + sim_loss
            # loss = loss + lambda_path * ib_path_loss + lambda_geno * ib_geno_loss
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
            
            risk, _ = _calculate_risk(h[0])

            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

            total_loss += loss_value 
            # if modality==  'coattn_motcat':
            #     loss = loss / 32 + 0
            loss.backward()
            optimizer.step()
            scheduler.step()

            window_loss_sum       += loss_value
            window_risk_sum       += weighted_risk_loss.item()
            window_sim_sum        += sim_loss.item()
            window_align_sum      += loss_align.item()
            window_kl_sum         += kl_align_loss.item()
            window_ib_path_sum    += weighted_ib_path_loss.item()
            window_ib_geno_sum    += weighted_ib_geno_loss.item()
            window_ib_fusion_sum  += weighted_ib_fusion_loss.item()
            # window_ib_path_active += ib_path_weight.mean().item()
            # window_ib_geno_active += ib_geno_weight.mean().item()
            # window_ib_fusion_active += ib_fusion_weight.mean().item()
            window_count += 1
                    # ========== 窗口结束条件：凑够 50 个样本，或 epoch 末尾剩余不足 50 的一块 ==========
            is_window_end = (window_count == log_every) or (
                batch_idx == num_batches - 1 and window_count > 0
            )
            if is_window_end:
                # 计算这一窗的平均
                w = float(window_count)
                avg_loss       = window_loss_sum       / w
                avg_risk       = window_risk_sum       / w
                avg_sim        = window_sim_sum        / w
                avg_align      = window_align_sum      / w
                avg_kl         = window_kl_sum         / w
                avg_ib_path    = window_ib_path_sum    / w
                avg_ib_geno    = window_ib_geno_sum    / w
                avg_ib_fusion  = window_ib_fusion_sum  / w
                # avg_path_act   = window_ib_path_active / w
                # avg_geno_act   = window_ib_geno_active / w
                # avg_fusion_act = window_ib_fusion_active / w

                # 控制台打印（每窗一次）
                print(
                    f"[Step {global_step}] (window size={window_count}) "
                    f"Loss={avg_loss:.4f} | "
                    f"Risk={avg_risk:.4f} | "
                    f"IB_path={avg_ib_path:.4f}, "
                    f"IB_geno={avg_ib_geno:.4f}, "
                    f"IB_fusion={avg_ib_fusion:.4f} | "
                    f"KL={avg_kl:.4f} | Align={avg_align:.4f} | Sim={avg_sim:.4f} | "
                    # f"Act[path={avg_path_act:.2f}, geno={avg_geno_act:.2f}, fusion={avg_fusion_act:.2f}]"
                )

                # 如果用 W&B，就把这一窗的平均记录进去
                if use_wandb and wandb_run is not None:
                    wandb_run.log({
                        "step": global_step,
                        "window_size": window_count,
                        "train_win/total_loss": avg_loss,
                        "train_win/risk_loss": avg_risk,
                        "train_win/sim_loss": avg_sim,
                        "train_win/align_loss": avg_align,
                        "train_win/kl_loss": avg_kl,
                        "train_win/ib_path_loss": avg_ib_path,
                        "train_win/ib_geno_loss": avg_ib_geno,
                        "train_win/ib_fusion_loss": avg_ib_fusion,
                    })
                 ## 改为tensorboard版本 现在改为只采用一个
                elif use_tensorboard and wandb_run is not None:
                    wandb_run.add_scalar("train_win/total_loss", avg_loss, global_step)
                    wandb_run.add_scalar("train_win/risk_loss", avg_risk, global_step)
                    wandb_run.add_scalar("train_win/sim_loss", avg_sim, global_step)
                    wandb_run.add_scalar("train_win/align_loss", avg_align, global_step)
                    wandb_run.add_scalar("train_win/kl_loss", avg_kl, global_step)
                    wandb_run.add_scalar("train_win/ib_path_loss", avg_ib_path, global_step)
                    wandb_run.add_scalar("train_win/ib_geno_loss", avg_ib_geno, global_step)
                    wandb_run.add_scalar("train_win/ib_fusion_loss", avg_ib_fusion, global_step)
                    wandb_run.add_scalar("train_win/window_size", window_count, global_step)  # 可选：记录窗口大小


                # 窗口清零，准备下一窗
                window_loss_sum = window_risk_sum = window_sim_sum = 0.0
                window_align_sum = window_kl_sum = 0.0
                window_ib_path_sum = window_ib_geno_sum = window_ib_fusion_sum = 0.0
                window_ib_path_active = window_ib_geno_active = window_ib_fusion_active = 0.0
                window_count = 0
            # if (batch_idx % 20) == 0:
            #     print(f"[Batch {batch_idx}] Loss={loss_value:.4f} | Risk={weighted_risk_loss.item():.4f} | IB=({weighted_ib_path_loss.item():.4f}, {weighted_ib_geno_loss.item():.4f}, {weighted_ib_fusion_loss.item():.4f}) | KL={kl_align_loss.item():.4f} | Align={loss_align.item():.4f} | Sim={sim_loss.item():.4f}")

            if (batch_idx % 20) == 0:
                weighted_ib_total = (
                    lambda_path * weighted_ib_path_loss.item() +
                    lambda_geno * weighted_ib_geno_loss.item() +
                    lambda_fusion * weighted_ib_fusion_loss.item()
                )
                print(f"[Batch {batch_idx}] "
                      f"Loss={loss_value:.4f} | "
                      f"Risk={weighted_risk_loss.item():.4f} | "
                      f"IB(λ*): {λ_ib * weighted_ib_total:.4f} "
                      f"[Path={λ_ib * lambda_path * weighted_ib_path_loss.item():.4f}, "
                      f"Geno={λ_ib * lambda_geno * weighted_ib_geno_loss.item():.4f}, "
                      f"Fusion={λ_ib * lambda_fusion * weighted_ib_fusion_loss.item():.4f}] | "
                      f"KL(λ={λ_kl})={λ_kl * kl_align_loss.item():.4f} | "
                      f"Align(λ={λ_align})={λ_align * loss_align.item():.4f} | "
                      f"Sim(λ={λ_sim})={λ_sim * sim_loss.item():.4f}")
            


        else:  
            h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data)
            if str(loss_fn)=='NLLSurvLoss()':
                loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor) 
            else:
                loss = loss_fn(
                        h, y_disc.to(h.device), censor.to(h.device)
                    )
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]
            
            risk, _ = _calculate_risk(h)

            all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)

            total_loss += loss_value 
            # if modality==  'coattn_motcat':
            #     loss = loss / 32 + 0
            loss.backward()
            optimizer.step()
            scheduler.step()

            if (batch_idx % 20) == 0:
                print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))

    return c_index, total_loss



def generate_eval_times(data, bins_original, eps=1e-4):
    """
    通用生成 which_times_to_eval_at:
    起点为 data.min() + eps，终点为 data.max() - eps，中间为 bins_original[1:-1]

    Parameters:
        data: np.ndarray or list, 所有 survival time（用于评估起止点）
        bins_original: np.ndarray[T+1], 用于分区的时间边界
        eps: float, 微偏移量避免边界问题

    Returns:
        which_times_to_eval_at: np.ndarray[T-1 + 2] → len = T+1
    """
    start = np.min(data) + eps
    end = np.max(data) - eps
    middle = bins_original[1:-2]  # 只保留“中间的”划分点
    return np.concatenate(([start], middle, [end]))



def clamp_survival_times_and_adjust_times(survival_train, survival_test, which_times_to_eval_at):
    max_train_time = np.max(survival_train['time'])
    survival_test_clamped = survival_test.copy()
    
    # 先截断生存时间，超过训练集最大时间的改成 max_train_time
    mask = survival_test_clamped['time'] > max_train_time
    if np.any(mask):
        print(f"{np.sum(mask)} 个测试样本生存时间超过训练集最大时间 {max_train_time}，将被截断。")
        survival_test_clamped['time'][mask] = max_train_time
    
    # 调整评估时间点，保证全部严格小于 max_train_time
    corrected_times = which_times_to_eval_at[which_times_to_eval_at < max_train_time - 1e-6]
    if len(corrected_times) < len(which_times_to_eval_at):
        corrected_times = np.append(corrected_times, max_train_time - 1e-6)
        print(f"评估时间点长度从 {len(which_times_to_eval_at)} 调整为 {len(corrected_times)}，追加末尾时间点保持覆盖区间")
        
    return survival_test_clamped, corrected_times




def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics 
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """
    
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins
    # which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])
    which_times_to_eval_at = generate_eval_times(data,bins_original)
    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc
   
    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    survival_test ,which_times_to_eval_at = clamp_survival_times_and_adjust_times(survival_train, survival_test,which_times_to_eval_at)
    # brier score 
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.
    # survival_test ,which_times_to_eval_at = clamp_survival_times_and_adjust_times(survival_train, survival_test,which_times_to_eval_at)
    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    # try:
    print("time:", which_times_to_eval_at[1])
    print("n_cases:", ((survival_test["event"] == 1) & (survival_test["time"] <= which_times_to_eval_at[1])).sum())
    print("n_controls:", ((survival_test["time"] > which_times_to_eval_at[1])).sum())
    # 初始化 valid_times 列表
    valid_times = []
    valid_idx = []  # 保存对应时间点的索引，用于切 estimate

    for i, t in enumerate(which_times_to_eval_at[1:]):  # 跳过第一个时间点
        n_cases = ((survival_test["event"] == 1) & (survival_test["time"] <= t)).sum()
        n_controls = (survival_test["time"] > t).sum()
        if n_cases > 0 and n_controls > 0:
            valid_times.append(t)
            valid_idx.append(i)  # 记录有效列索引

    # 切 estimate，使列数与 valid_times 对应
    estimate_valid = 1 - all_risk_by_bin_scores[:, 1:][:, valid_idx]

    # 计算 cumulative_dynamic_auc
    _, iauc = cumulative_dynamic_auc(
        survival_train, 
        survival_test,
        estimate=estimate_valid, 
        times=valid_times
    )

    # _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    # except:
    #     print('An error occured while computing iauc')
    #     iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc

def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None,test_modality="path_and_geno"):
    r"""
    Run a validation loop on the trained model 
    
    Args: 
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
        - survival_train : np.array
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    all_risk_scores = []
    all_risk_by_bin_scores = []
    all_censorships = []
    all_event_times = []
    all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    with torch.no_grad():
        for data in loader:
            if modality in ['BezierSurv']:
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask,patient_id= _unpack_data(modality, device, data)
            else:
                data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)

            if modality in ["BezierSurv"]:
                input_args = {"x_path": data_WSI.to(device)}
                input_args["x_omic"] = data_omics.to(device)
                input_args["label"] = y_disc
                input_args['censor'] = censor
                input_args['training'] = False
                input_args['input_modality'] = test_modality
                input_args['return_feature'] = False
                input_args['event_time']=event_time
                    # ===== 只在第一次进入这个分支时计算 FLOPs & 参数量 =====
                if not hasattr(model, "_flops_done"):
                    import torch.nn as nn
                    from thop import profile

                    class FlopsWrapper(nn.Module):
                        def __init__(self, m):
                            super().__init__()
                            self.m = m

                        def forward(self, x_path, x_omic):
                            # 这里固定和上面 input_args 一致的 kwargs
                            return self.m(
                                x_path=x_path,
                                x_omic=x_omic,
                                label=None,          # 计算 FLOPs 不需要 label / censor
                                censor=None,
                                training=False,
                                input_modality=test_modality,
                                return_feature=False,
                                event_time=None,
                            )

                    wrapper = FlopsWrapper(model).to(device)
                    wrapper.eval()
                    with torch.no_grad():
                        flops, params = profile(
                            wrapper,
                            inputs=(input_args["x_path"], input_args["x_omic"]),
                            verbose=False,
                        )

                    print(f"[FLOPs] params = {params} ({params/1e6:.2f} M)")
                    print(f"[FLOPs] per forward = {flops:.3e} FLOPs ({flops/1e9:.2f} GFLOPs)")

                    # 打个标记，后面就不再算了
                    model._flops_done = True

                h = model(**input_args)
                h = h[0]
            
            else:
                h = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI, 
                    mask = mask
                    )
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            
            if str(loss_fn)=='NLLSurvLoss()':
                loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            else:
                loss = loss_fn(
                h, y_disc.to(h.device), censor.to(h.device)
                     )
            loss_value = loss.item()
            loss = loss / y_disc.shape[0]


            risk, risk_by_bin = _calculate_risk(h)
            all_risk_by_bin_scores.append(risk_by_bin)
            all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            all_slide_ids.append(slide_ids.values[count])
            count += 1

    total_loss /= len(loader.dataset)
    all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    all_censorships = np.concatenate(all_censorships, axis=0)
    all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        patient_results[case_id]["time"] = all_event_times[i]
        patient_results[case_id]["risk"] = all_risk_scores[i]
        patient_results[case_id]["censorship"] = all_censorships[i]
        patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss


def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs

    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    return lr_scheduler

def log_print(f_name,str_text):
    with open(f_name,'a') as f:
        f.write(str(str_text)+'\n')

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
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

def compute_classwise_gmm_center(mu_all, std_all, proto_pi):
    """
    计算每类 GMM 的整体协方差（均值不做权重加权）

    Args:
        mu_all:    [C, K, D] 每类每个分量的均值
        std_all:   [C, K, D] 每类每个分量的标准差
        proto_pi:  [C, K]    每类 GMM 的分量权重

    Returns:
        mu_center: [C, D] 每类的中心直接取均值
        covs:      [C, D, D] 每类整体协方差
    """
    C, K, D = mu_all.shape
    mu_center = mu_all.mean(dim=1)  # 不加权中心 [C, D]
    cov_all = []

    for c in range(C):
        pi = proto_pi[c]  # [K]
        mu = mu_all[c]    # [K, D]
        std = std_all[c]  # [K, D]
        center = mu_center[c]  # [D]

        cov = torch.zeros(D, D, device=mu.device)
        for k in range(K):
            diff = mu[k] - center  # [D]
            outer = torch.outer(diff, diff)  # [D, D]
            local_cov = torch.diag(std[k] ** 2)  # [D, D]
            cov += pi[k] * (outer + local_cov)

        cov_all.append(cov)

    return mu_center, torch.stack(cov_all)  # [C, D], [C, D, D]

def generate_bezier_global_path(mu_endpoints, bezier_ctrl, proto_pi, bezier_curve_fn, n_classes):
    C, K = proto_pi.shape
    D = mu_endpoints.shape[-1]

    pi_0 = torch.softmax(proto_pi[0], dim=-1).view(1, K)
    pi_1 = torch.softmax(proto_pi[-1], dim=-1).view(1, K)

    mu0 = (mu_endpoints[0] * pi_0.T)
    mu1 = (mu_endpoints[1] * pi_1.T)

    ctrl_seq = torch.cat([
        mu0.unsqueeze(0),
        bezier_ctrl,
        mu1.unsqueeze(0)
    ], dim=0)

    mu_list = []
    for c in range(n_classes):
        t = c / (n_classes - 1)
        mu = bezier_curve_fn(t, ctrl_seq)
        mu_list.append(mu)

    return torch.stack(mu_list, dim=0)  # [C, D]

from scipy.stats import multivariate_normal
from matplotlib import cm

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.stats import multivariate_normal

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from matplotlib import cm
from matplotlib.patches import Ellipse


def gmm_reparameterize_noise_only(mu: torch.Tensor, std: torch.Tensor, pi_logits: torch.Tensor, temperature=1.0):
    """
    Args:
        mu: [D] or [1, D] — 固定的整体 GMM 均值（来自贝塞尔路径）
        std: [K, D] — 每个分量的标准差
        pi_logits: [K] — 每个分量的权重 logits
        temperature: float — Gumbel softmax 温度

    Returns:
        sample: [D] — 最终 Gumbel 重参数化后的 GMM 采样结果
    """
    K, D = std.shape
    std = F.softplus(std)

    # 处理异常 logits（NaN/Inf）
    if torch.isnan(pi_logits).any() or torch.isinf(pi_logits).any():
        pi_logits = torch.where(torch.isnan(pi_logits), torch.zeros_like(pi_logits), pi_logits)
        pi_logits = torch.clamp(pi_logits, -20, 20)
    # torch.manual_seed(42)
    # Gumbel-softmax 权重
    gumbel = F.gumbel_softmax(pi_logits, tau=max(temperature, 1e-3), hard=False)  # [K]

    # 重参数化噪声项
    eps = torch.randn_like(std)  # [K, D]
    noise = std * eps            # [K, D]
    weighted_noise = torch.sum(gumbel[:, None] * noise, dim=0)  # [D]

    # 输出：共享均值 + Gumbel 噪声扰动
    mu = mu.squeeze(0) if mu.dim() == 2 else mu  # ensure [D]
    sample = mu + weighted_noise  # [D]

    return sample

from scipy.stats import gaussian_kde


    


import numpy as np
import torch
import torch.nn.functional as F


def _step0731(cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader,test_loader = None):
    r"""
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """
    ## wandb 版本 有点麻烦
    if args.use_wandb:

        import wandb
        
        wandb.login(key="自己的key")

        config = {
            "model": args.modality,
            "study": args.study,
            "lr": args.lr,
            "epochs": args.max_epochs,
            "batch_size": args.batch_size,
            "num_instances_per_slide": args.num_patches,
            "fold": cur,
            # 你还可以加：GPU count, dataset_size 等
        }
        run_name = "modality_{}_dataset_{}_fold_{}".format(
            args.modality, args.study, cur
        )
        # 'tcga_blca__nclass8_sp_l2Weight_1e-05_our_b2_survival_months_dss_dim1_1024_patches_4000_wsiDim_256_epochs_75_fusion_concat_modality_BezierSurv_pathT_combine_warmup_1_ctrlp_3_seed_1_mil_TransMIL_IBP_IBG_Align_Sim_spl_c'
        run_id = "modality_{}_dataset_{}_{}_{}_fold{}".format(args.modality, args.study,args.prototype_mode,args.results_dir.split('/')[-1][-20:], cur)  # 用来控制目录后缀

        run = wandb.init(
            project="BezierSurv",
            config=config,
            name=run_id,   # UI 和目录里都会看到这个名字
            id=run_id,       # 离线目录会叫：offline-run-时间-这个id
            reinit=True,     # 同一进程里多次 init（多折）需要这个
            mode="offline",  # 已经用环境变量设了 WANDB_MODE，可以不写；想写也可以
            # entity="your_team_name",
        )


        wandb.config.update(config)
       
    
         


    # -------- TensorBoard 初始化 --------
    elif args.use_tensorboard:

        from torch.utils.tensorboard import SummaryWriter
        run_name = "modality_{}_dataset_{}_fold_{}".format(args.modality, args.study, cur)
        log_dir = os.path.join(args.results_dir, "runs", run_name)  # 日志保存目录
        run = SummaryWriter(log_dir=log_dir)
    else:
        run = None


    flags = []
    # if args.use_bezier_gmm: flags.append("bezier")
    if args.use_ib_path: flags.append("ibpath")
    if args.use_ib_geno: flags.append("ibgeno")
    if args.use_ib_fusion: flags.append("ibfusion")
    if args.use_align_loss: flags.append("align")
    if args.use_sim_loss: flags.append("sim")
    if args.use_kl_align: flags.append("kl")
    if args.use_self_paced_ablation: flags.append("zibuguanbi")
    

    # flag_str = "_".join(flags) if flags else "base"

    # f_name = f"{args.results_dir}_{flag_str}_fold_{cur}.log"
    f_name = '{}_fold_{}.log'.format(args.results_dir,cur)
    log_print(f_name,"=======================FOLD {}=======================".format(cur))

    all_survival = _extract_survival_metadata(train_loader, val_loader)
    best_val_cindex = 0
    best_val_loss = float('inf') 
    best_val_IBS = float('inf') 
    best_val_iauc = 0
    early_stop_counter = 0 ## 设置早停机制
    best_ckpts = []     # 存储 dict：{ epoch, val_cindex, val_loss, test_cindex, test_loss }
    topk = 3  ## 挑选三个最好的模型进行测试 取平均
    early_stop_patience = args.early_stop_patience
    difficulty_buffer = {}  # 全局字典：sample_id -> {"loss": val, "ib_path": val, ...}

    for epoch in range(args.max_epochs):

            if args.modality in ["BezierSurv"]:

                ## 修改一下自步学习的方式
                if args.use_self_paced:
                    if epoch % 2 ==0:
                        difficulty_buffer = update_difficulty_buffer(model, train_loader, args.modality, loss_fn, difficulty_buffer)
                    if args.use_wandb or args.use_tensorboard:
                        train_cindex, train_total_loss=wandb_train_loop_survival_self_paced(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn,difficulty_buffer,
                                                                                    use_self_paced_ablation=args.use_self_paced_ablation,warmup_epochs=args.warmup_epochs,
                                                                                    spl_on_censored_only=args.spl_on_censored_only,
                                                                                    use_modal_balance=args.use_modal_balance,
                                                                                    wandb_run=run,
                                                                                    use_wandb=args.use_wandb,
                                                                                    use_tensorboard=args.use_tensorboard)

                    else:
                        train_cindex, train_total_loss=_train_loop_survival_self_paced(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn,difficulty_buffer,
                                                                                    use_self_paced_ablation=args.use_self_paced_ablation,warmup_epochs=args.warmup_epochs,
                                                                                    spl_on_censored_only=args.spl_on_censored_only,use_modal_balance=args.use_modal_balance)
                    
                else:
                    train_cindex, train_total_loss=_train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)

                if epoch >0 and epoch %1 ==0:  ## 为了提速 每3个peoch 进行测试

                    
                    if args.test_all_modalities:  
                        # Check if validation data exists

                        if test_loader is None:  # 如果没有测试集数据
                            test_patient_results_path = test_cindex_path = test_cindex_ipcw_path = test_BS_path = test_IBS_path = test_iauc_path = test_total_loss_path = 0
                            test_patient_results_geno = test_cindex_geno = test_cindex_ipcw_geno = test_BS_geno = test_IBS_geno = test_iauc_geno = test_total_loss_geno = 0
                            test_patient_results_path_geno = test_cindex_path_geno = test_cindex_ipcw_path_geno = test_BS_path_geno = test_IBS_path_geno = test_iauc_path_geno = test_total_loss_path_geno = 0
                        else:
                            test_patient_results_path, test_cindex_path, test_cindex_ipcw_path, test_BS_path, test_IBS_path, test_iauc_path, test_total_loss_path = _summary(
                                args.dataset_factory, model, args.modality, test_loader, loss_fn, all_survival, test_modality="path")

                            test_patient_results_geno, test_cindex_geno, test_cindex_ipcw_geno, test_BS_geno, test_IBS_geno, test_iauc_geno, test_total_loss_geno = _summary(
                                args.dataset_factory, model, args.modality, test_loader, loss_fn, all_survival, test_modality="geno")

                            test_patient_results_path_geno, test_cindex_path_geno, test_cindex_ipcw_path_geno, test_BS_path_geno, test_IBS_path_geno, test_iauc_path_geno, test_total_loss_path_geno = _summary(
                                args.dataset_factory, model, args.modality, test_loader, loss_fn, all_survival, test_modality="path_and_geno")

                        # Test: Path + Genomic
                        results, val_cindex_path_geno, val_cindex_ipcw_path_geno, val_BS_path_geno, val_IBS_path_geno, val_iauc_path_geno, val_total_loss_path_geno = _summary(
                            args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, test_modality="path_and_geno")

                        # Validation: Path
                        val_patient_results_path, val_cindex_path, val_cindex_ipcw_path, val_BS_path, val_IBS_path, val_iauc_path, val_total_loss_path = _summary(
                            args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, test_modality="path")

                        # Validation: Genomic
                        val_patient_results_geno, val_cindex_geno, val_cindex_ipcw_geno, val_BS_geno, val_IBS_geno, val_iauc_geno, val_total_loss_geno = _summary(
                            args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, test_modality="geno")

                        # Calculate average metrics for validation and test
                        val_cindex = (val_cindex_path_geno + val_cindex_path + val_cindex_geno) / 3 if val_loader else 0
                        val_cindex_ipcw = (val_cindex_ipcw_path_geno + val_cindex_ipcw_path + val_cindex_ipcw_geno) / 3 if val_loader else 0
                        val_BS = (val_BS_path_geno + val_BS_path + val_BS_geno) / 3 if val_loader else 0
                        val_IBS = (val_IBS_path_geno + val_IBS_path + val_IBS_geno) / 3 if val_loader else 0
                        val_iauc = (val_iauc_path_geno + val_iauc_path + val_iauc_geno) / 3 if val_loader else 0
                        val_total_loss = (val_total_loss_path_geno + val_total_loss_path + val_total_loss_geno) / 3 if val_loader else 0

                        test_cindex = (test_cindex_path_geno + test_cindex_path + test_cindex_geno) / 3
                        test_cindex_ipcw = (test_cindex_ipcw_path_geno + test_cindex_ipcw_path + test_cindex_ipcw_geno) / 3
                        test_BS = (test_BS_path_geno + test_BS_path + test_BS_geno) / 3
                        test_IBS = (test_IBS_path_geno + test_IBS_path + test_IBS_geno) / 3
                        test_iauc = (test_iauc_path_geno + test_iauc_path + test_iauc_geno) / 3
                        test_total_loss = (test_total_loss_path_geno + test_total_loss_path + test_total_loss_geno) / 3

                        # === Print results ===
                        # Validation results
                        print(f"[Val] path_and_geno: loss={val_total_loss_path_geno}, c-index={val_cindex_path_geno}, c-index_ipcw={val_cindex_ipcw_path_geno}, BS={val_BS_path_geno}, IBS={val_IBS_path_geno}, iauc={val_iauc_path_geno}")
                        print(f"[Val] path:           loss={val_total_loss_path}, c-index={val_cindex_path}, c-index_ipcw={val_cindex_ipcw_path}, BS={val_BS_path}, IBS={val_IBS_path}, iauc={val_iauc_path}")
                        print(f"[Val] geno:           loss={val_total_loss_geno}, c-index={val_cindex_geno}, c-index_ipcw={val_cindex_ipcw_geno}, BS={val_BS_geno}, IBS={val_IBS_geno}, iauc={val_iauc_geno}")
                        print(f"[Val] avg:            loss={val_total_loss}, c-index={val_cindex}, c-index_ipcw={val_cindex_ipcw}, BS={val_BS}, IBS={val_IBS}, iauc={val_iauc}")

                        # Test results
                        print(f"[Test] path_and_geno: loss={test_total_loss_path_geno}, c-index={test_cindex_path_geno}, c-index_ipcw={test_cindex_ipcw_path_geno}, BS={test_BS_path_geno}, IBS={test_IBS_path_geno}, iauc={test_iauc_path_geno}")
                        print(f"[Test] path:           loss={test_total_loss_path}, c-index={test_cindex_path}, c-index_ipcw={test_cindex_ipcw_path}, BS={test_BS_path}, IBS={test_IBS_path}, iauc={test_iauc_path}")
                        print(f"[Test] geno:           loss={test_total_loss_geno}, c-index={test_cindex_geno}, c-index_ipcw={test_cindex_ipcw_geno}, BS={test_BS_geno}, IBS={test_IBS_geno}, iauc={test_iauc_geno}")
                        print(f"[Test] avg:            loss={test_total_loss}, c-index={test_cindex}, c-index_ipcw={test_cindex_ipcw}, BS={test_BS}, IBS={test_IBS}, iauc={test_iauc}")

                        # === 日志记录 ===
                        log_print(f_name, f"----------Epoch {epoch}----------")
                        log_print(f_name, f"train cindex: {train_cindex}, train loss: {train_total_loss}")

                        log_print(f_name, f"[Val] path_and_geno: loss={val_total_loss_path_geno}, c-index={val_cindex_path_geno}, c-index_ipcw={val_cindex_ipcw_path_geno}, BS={val_BS_path_geno}, IBS={val_IBS_path_geno}, iauc={val_iauc_path_geno}")
                        log_print(f_name, f"[Val] path:           loss={val_total_loss_path}, c-index={val_cindex_path}, c-index_ipcw={val_cindex_ipcw_path}, BS={val_BS_path}, IBS={val_IBS_path}, iauc={val_iauc_path}")
                        log_print(f_name, f"[Val] geno:           loss={val_total_loss_geno}, c-index={val_cindex_geno}, c-index_ipcw={val_cindex_ipcw_geno}, BS={val_BS_geno}, IBS={val_IBS_geno}, iauc={val_iauc_geno}")
                        log_print(f_name, f"[Val] avg:            loss={val_total_loss}, c-index={val_cindex}, c-index_ipcw={val_cindex_ipcw}, BS={val_BS}, IBS={val_IBS}, iauc={val_iauc}")

                        log_print(f_name, f"[Test] path_and_geno: loss={test_total_loss_path_geno}, c-index={test_cindex_path_geno}, c-index_ipcw={test_cindex_ipcw_path_geno}, BS={test_BS_path_geno}, IBS={test_IBS_path_geno}, iauc={test_iauc_path_geno}")
                        log_print(f_name, f"[Test] path:           loss={test_total_loss_path}, c-index={test_cindex_path}, c-index_ipcw={test_cindex_ipcw_path}, BS={test_BS_path}, IBS={test_IBS_path}, iauc={test_iauc_path}")
                        log_print(f_name, f"[Test] geno:           loss={test_total_loss_geno}, c-index={test_cindex_geno}, c-index_ipcw={test_cindex_ipcw_geno}, BS={test_BS_geno}, IBS={test_IBS_geno}, iauc={test_iauc_geno}")
                        log_print(f_name, f"[Test] avg:            loss={test_total_loss}, c-index={test_cindex}, c-index_ipcw={test_cindex_ipcw}, BS={test_BS}, IBS={test_IBS}, iauc={test_iauc}")

                        data_save = [
                            epoch,

                            # 测试集平均指标（含 IPCW）
                            test_cindex,
                            test_cindex_ipcw,
                            test_IBS,
                            test_BS,
                            test_iauc,

                            # 测试集详细指标，按 path_geno -> path -> geno 顺序，含 IPCW
                            test_cindex_path_geno,
                            test_cindex_path,
                            test_cindex_geno,

                            test_cindex_ipcw_path_geno,
                            test_cindex_ipcw_path,
                            test_cindex_ipcw_geno,

                            test_IBS_path_geno,
                            test_IBS_path,
                            test_IBS_geno,

                            test_BS_path_geno,
                            test_BS_path,
                            test_BS_geno,

                            test_iauc_path_geno,
                            test_iauc_path,
                            test_iauc_geno,

                            # 验证集平均指标（含 IPCW）
                            val_cindex,
                            val_cindex_ipcw,
                            val_IBS,
                            val_BS,
                            val_iauc,

                            # 验证集详细指标，按 path_geno -> path -> geno 顺序，含 IPCW
                            val_cindex_path_geno,
                            val_cindex_path,
                            val_cindex_geno,

                            val_cindex_ipcw_path_geno,
                            val_cindex_ipcw_path,
                            val_cindex_ipcw_geno,

                            val_IBS_path_geno,
                            val_IBS_path,
                            val_IBS_geno,

                            val_BS_path_geno,
                            val_BS_path,
                            val_BS_geno,

                            val_iauc_path_geno,
                            val_iauc_path,
                            val_iauc_geno,

                            # 损失值（若需要）
                            val_total_loss_path_geno,
                            val_total_loss_path,
                            val_total_loss_geno,
                            test_total_loss_path_geno,
                            test_total_loss_path,
                            test_total_loss_geno,
                        ]

                    
                    else:
                        

                        _, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, val_total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival, test_modality="path_and_geno") 
                        _, test_cindex, test_cindex_ipcw, test_BS, test_IBS, test_iauc, test_total_loss = _summary(args.dataset_factory, model, args.modality, test_loader, loss_fn, all_survival,test_modality="path_and_geno") if test_loader else (0, 0, 0, 0, 0, 0, 0)

                        # 打印验证集指标，如果没有验证集数据，输出 0
                        print(f"[Val] loss: {val_total_loss}, c-index: {val_cindex}, c-index_ipcw: {val_cindex_ipcw}, BS: {val_BS}, IBS: {val_IBS}, iauc: {val_iauc}")

                        # 打印测试指标
                        print(f"[Test] loss: {test_total_loss}, c-index: {test_cindex}, c-index_ipcw: {test_cindex_ipcw}, BS: {test_BS}, IBS: {test_IBS}, iauc: {test_iauc}")

                        # 日志记录
                        log_print(f_name, f"----------Epoch {epoch}----------")
                        log_print(f_name, f"train cindex: {train_cindex} train loss: {train_total_loss}")
                        log_print(f_name, f"val loss: {val_total_loss}, val c-index: {val_cindex}, val c-index_ipcw: {val_cindex_ipcw}, val BS: {val_BS}, val IBS: {val_IBS}, val iauc: {val_iauc}")
                        log_print(f_name, f"test loss: {test_total_loss}, test c-index: {test_cindex}, test c-index_ipcw: {test_cindex_ipcw}, test BS: {test_BS}, test IBS: {test_IBS}, test iauc: {test_iauc}")

                        # 数据存储列表
                        data_save = [
                            epoch,
                            val_cindex ,
                            val_cindex_ipcw ,
                            val_BS ,
                            val_IBS ,
                            val_iauc,
                            1, 1, 1, 1, 1, 1,  # placeholder values
                            1, 1, 1, 1, 1, 1,  # placeholder values
                            1, 1, 1,

                            test_cindex if test_loader else 0,
                            test_cindex_ipcw if test_loader else 0,
                            test_BS if test_loader else 0,
                            test_IBS if test_loader else 0,
                            test_iauc if test_loader else 0,
                            1, 1, 1, 1, 1, 1,  # placeholder values
                            1, 1, 1, 1, 1, 1,  # placeholder values
                            1, 1, 1,
                            1, 1, 1, 1, 1, 1  # placeholder values
                        ]
                   


                    after_warmup = epoch >= args.warmup_epochs
                    cindex_threshold = 0.015         # C-index 最小提升阈值
                    loss_improvement_threshold = 0.01  # 损失至少下降 1%
                    ibs_improvement_threshold = 0.01   # IBS 至少下降 1%

                    if args.use_wandb and run is not None:
                        log_dict = {
                            "epoch": epoch,

                            # ================= train =================
                            "train/c_index":      train_cindex,
                            "train/total_loss":   train_total_loss,

                            # ================= test: 平均指标 =================
                            "test/avg_c_index":      test_cindex,
                            "test/avg_c_index_ipcw": test_cindex_ipcw,
                            "test/avg_IBS":          test_IBS,
                            "test/avg_BS":           test_BS,
                            "test/avg_iAUC":         test_iauc,

                            # ================= test: 分模态 =================
                            # path+geno 联合
                            "test/path_geno/c_index":      test_cindex_path_geno,
                            "test/path_geno/c_index_ipcw": test_cindex_ipcw_path_geno,
                            "test/path_geno/IBS":          test_IBS_path_geno,
                            "test/path_geno/BS":           test_BS_path_geno,
                            "test/path_geno/iAUC":         test_iauc_path_geno,
                            "test/path_geno/total_loss":   test_total_loss_path_geno,

                            # 只用 path
                            "test/path/c_index":      test_cindex_path,
                            "test/path/c_index_ipcw": test_cindex_ipcw_path,
                            "test/path/IBS":          test_IBS_path,
                            "test/path/BS":           test_BS_path,
                            "test/path/iAUC":         test_iauc_path,
                            "test/path/total_loss":   test_total_loss_path,

                            # 只用 geno
                            "test/geno/c_index":      test_cindex_geno,
                            "test/geno/c_index_ipcw": test_cindex_ipcw_geno,
                            "test/geno/IBS":          test_IBS_geno,
                            "test/geno/BS":           test_BS_geno,
                            "test/geno/iAUC":         test_iauc_geno,
                            "test/geno/total_loss":   test_total_loss_geno,

                            # ================= val: 平均指标 =================
                            "val/avg_c_index":      val_cindex,
                            "val/avg_c_index_ipcw": val_cindex_ipcw,
                            "val/avg_IBS":          val_IBS,
                            "val/avg_BS":           val_BS,
                            "val/avg_iAUC":         val_iauc,

                            # ================= val: 分模态 =================
                            # path+geno 联合
                            "val/path_geno/c_index":      val_cindex_path_geno,
                            "val/path_geno/c_index_ipcw": val_cindex_ipcw_path_geno,
                            "val/path_geno/IBS":          val_IBS_path_geno,
                            "val/path_geno/BS":           val_BS_path_geno,
                            "val/path_geno/iAUC":         val_iauc_path_geno,
                            "val/path_geno/total_loss":   val_total_loss_path_geno,

                            # 只用 path
                            "val/path/c_index":      val_cindex_path,
                            "val/path/c_index_ipcw": val_cindex_ipcw_path,
                            "val/path/IBS":          val_IBS_path,
                            "val/path/BS":           val_BS_path,
                            "val/path/iAUC":         val_iauc_path,
                            "val/path/total_loss":   val_total_loss_path,

                            # 只用 geno
                            "val/geno/c_index":      val_cindex_geno,
                            "val/geno/c_index_ipcw": val_cindex_ipcw_geno,
                            "val/geno/IBS":          val_IBS_geno,
                            "val/geno/BS":           val_BS_geno,
                            "val/geno/iAUC":         val_iauc_geno,
                            "val/geno/total_loss":   val_total_loss_geno,
                        }

                        run.log(log_dict)
                        print('Epoch: {}, val_loss: {:.4f},val_c_index: {:.4f}'.format(epoch, test_total_loss, test_cindex))
                    # # 各项改进判断
                    elif args.use_tensorboard and run is not None:
                        # ============ Train ============
                        run.add_scalar("train/c_index", train_cindex, epoch)
                        run.add_scalar("train/total_loss", train_total_loss, epoch)

                        # ============ Test: avg ============
                        run.add_scalar("test/avg_c_index", test_cindex, epoch)
                        run.add_scalar("test/avg_c_index_ipcw", test_cindex_ipcw, epoch)
                        run.add_scalar("test/avg_IBS", test_IBS, epoch)
                        # run.add_scalar("test/avg_BS", test_BS, epoch)
                        run.add_scalar("test/avg_iAUC", test_iauc, epoch)

                        # ============ Test: Path + Geno ============
                        run.add_scalar("test/path_geno/c_index", test_cindex_path_geno, epoch)
                        run.add_scalar("test/path_geno/c_index_ipcw", test_cindex_ipcw_path_geno, epoch)
                        run.add_scalar("test/path_geno/IBS", test_IBS_path_geno, epoch)
                        # run.add_scalar("test/path_geno/BS", test_BS_path_geno, epoch)
                        run.add_scalar("test/path_geno/iAUC", test_iauc_path_geno, epoch)
                        run.add_scalar("test/path_geno/total_loss", test_total_loss_path_geno, epoch)

                        # ============ Test: Path only ============
                        run.add_scalar("test/path/c_index", test_cindex_path, epoch)
                        run.add_scalar("test/path/c_index_ipcw", test_cindex_ipcw_path, epoch)
                        run.add_scalar("test/path/IBS", test_IBS_path, epoch)
                        # run.add_scalar("test/path/BS", test_BS_path, epoch)
                        run.add_scalar("test/path/iAUC", test_iauc_path, epoch)
                        run.add_scalar("test/path/total_loss", test_total_loss_path, epoch)

                        # ============ Test: Geno only ============
                        run.add_scalar("test/geno/c_index", test_cindex_geno, epoch)
                        run.add_scalar("test/geno/c_index_ipcw", test_cindex_ipcw_geno, epoch)
                        run.add_scalar("test/geno/IBS", test_IBS_geno, epoch)
                        # run.add_scalar("test/geno/BS", test_BS_geno, epoch)
                        run.add_scalar("test/geno/iAUC", test_iauc_geno, epoch)
                        run.add_scalar("test/geno/total_loss", test_total_loss_geno, epoch)

                        # ============ Val: avg ============
                        run.add_scalar("val/avg_c_index", val_cindex, epoch)
                        run.add_scalar("val/avg_c_index_ipcw", val_cindex_ipcw, epoch)
                        run.add_scalar("val/avg_IBS", val_IBS, epoch)
                        # run.add_scalar("val/avg_BS", val_BS, epoch)
                        run.add_scalar("val/avg_iAUC", val_iauc, epoch)

                        # ============ Val: Path + Geno ============
                        run.add_scalar("val/path_geno/c_index", val_cindex_path_geno, epoch)
                        run.add_scalar("val/path_geno/c_index_ipcw", val_cindex_ipcw_path_geno, epoch)
                        run.add_scalar("val/path_geno/IBS", val_IBS_path_geno, epoch)
                        # run.add_scalar("val/path_geno/BS", val_BS_path_geno, epoch)
                        run.add_scalar("val/path_geno/iAUC", val_iauc_path_geno, epoch)
                        run.add_scalar("val/path_geno/total_loss", val_total_loss_path_geno, epoch)

                        # ============ Val: Path only ============
                        run.add_scalar("val/path/c_index", val_cindex_path, epoch)
                        run.add_scalar("val/path/c_index_ipcw", val_cindex_ipcw_path, epoch)
                        run.add_scalar("val/path/IBS", val_IBS_path, epoch)
                        # run.add_scalar("val/path/BS", val_BS_path, epoch)
                        run.add_scalar("val/path/iAUC", val_iauc_path, epoch)
                        run.add_scalar("val/path/total_loss", val_total_loss_path, epoch)

                        # ============ Val: Geno only ============
                        run.add_scalar("val/geno/c_index", val_cindex_geno, epoch)
                        run.add_scalar("val/geno/c_index_ipcw", val_cindex_ipcw_geno, epoch)
                        run.add_scalar("val/geno/IBS", val_IBS_geno, epoch)
                        # run.add_scalar("val/geno/BS", val_BS_geno, epoch)
                        run.add_scalar("val/geno/iAUC", val_iauc_geno, epoch)
                        run.add_scalar("val/geno/total_loss", val_total_loss_geno, epoch)

                        run.add_scalar("val/total_loss", val_total_loss, epoch)
                        run.add_scalar("val/val_cindex", val_cindex, epoch)
                        
                        print(f'Epoch: {epoch}, val_loss: {val_total_loss:.4f}, val_c_index: {val_cindex:.4f}')

                    cindex_improved = (val_cindex - best_val_cindex) > cindex_threshold
                    loss_improved = val_total_loss < best_val_loss * (1 - loss_improvement_threshold)
                    ibs_improved = val_IBS < best_val_IBS * (1 - ibs_improvement_threshold)

                    # 加权权重（可根据实际调整）
                    alpha = 1.0   # C-index 权重（越大越好）
                    beta = 0.1    # Loss 权重（越小越好，loss下降为正贡献）
                    gamma = 0.5   # IBS 权重（越小越好，IBS下降为正贡献）

                    # 计算综合得分，注意test_IBS越小越好，所以用best_test_IBS - test_IBS，转成“提升”指标
                    # improvement_score = alpha * (val_cindex - best_val_cindex) + beta * (best_val_loss - val_total_loss) + gamma * (best_val_IBS - val_IBS)
                    # combined_score = 0.5 * test_cindex + 0.5 * test_iauc
                    # improvement_score = alpha * (val_cindex - best_val_cindex) + beta * (best_val_loss - val_total_loss) + gamma * (val_iauc - best_val_iauc)
                    if args.prototype_mode =='memory_bank':
                        improvement_score = alpha * (val_cindex - best_val_cindex) +  1 * (val_iauc - best_val_iauc)+ 0.1 * (best_val_IBS - val_IBS)
                    else:
                        improvement_score = alpha * (val_cindex - best_val_cindex) +  gamma * (val_iauc - best_val_iauc)+ gamma * (best_val_IBS - val_IBS)
                    
                    # 综合阈值，需综合考虑调参
                    improvement_threshold = 0.0001

                    if improvement_score > improvement_threshold:
                        early_stop_counter = 0  # 重置早停计数器
                        
                        # 更新最好的测试集结果
                        best_val_loss = val_total_loss
                        best_val_cindex = val_cindex
                        best_val_IBS = val_IBS
                        best_val_iauc = val_iauc
                        log_print(f_name, f"best: epoch={epoch}")

                        # 保存当前折的最佳测试集结果
                        result_save_fold = data_save  # 假设 data_save 已正确赋值
                        
                        # 保存最好的模型
                        save_path = os.path.join(
                            args.results_dir,
                            "s_fold-{}_epoch-{}_testcindex-{:.4f}_valcindex-{:.4f}_testIauc-{:.4f}_val_Iauc-{:.4f}.pt".format(
                                cur, epoch, test_cindex, val_cindex, test_iauc, val_iauc  # 没有验证集，设置为 0
                            )
                        )
                        torch.save(model.state_dict(), save_path)

                        # 可视化

                            

                # pass

           
    
    # if args.modality not in ["BezierSurv"]:
    #     results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.modality, test_loader, loss_fn, all_survival)
    
    print('Final Test c-index: {:.4f}'.format(test_cindex))
    # print('Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
    #     val_cindex, 
    #     val_cindex_ipcw,
    #     val_IBS,
    #     val_iauc
    #     ))

    # return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)
    ## 修改为如果是我们自己的模型 先不计算其他指标
    if args.modality in ["BezierSurv"]:
        return None, (None, None, None, None, None, None),result_save_fold
    else:
        return None, (None, None, None, None, None, None),result_save_fold

def get_model_size(model):
    """返回模型参数量，并打印成可直接填表的格式"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def human_readable(n):
        if n >= 1e9:
            return f"{n/1e9:.2f}B"
        elif n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.1f}K"
        else:
            return str(n)

    print(f"[Model] total params      : {total_params} ({human_readable(total_params)})")
    print(f"[Model] trainable params  : {trainable_params} ({human_readable(trainable_params)})")

    # 表单一般填“可训练参数量”
    model_size_str = f"{human_readable(trainable_params)} parameters"
    return trainable_params, model_size_str


def _train_val_0731(datasets, cur, args):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int 
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    #----> gets splits and summarize
    if args.val_test:
        train_split, val_split, test_split = _get_splits(datasets, cur, args)
    else:
        train_split, val_split = _get_splits(datasets, cur, args)
    
    #----> init loss function
    loss_fn = _init_loss_function(args)

    #----> init model
    model = _init_model(args)
    num_params, model_size_str = get_model_size(model)
    print(">> Fill in the form with model size:", model_size_str) 
    #---> init optimizer
    optimizer = _init_optim(args, model)

    #---> init loaders
    if args.val_test:
        train_loader, val_loader, test_loader = _init_loaders(args, train_split, val_split, test_split)
    else:
        train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # lr scheduler 
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    #---> do train val

    if args.val_test:
        ''' 
        result_save_fold = [epoch, val_cindex_path, val_cindex_geno, val_cindex_path_geno,val_cindex,
                        test_cindex_path_geno, test_cindex_path, test_cindex_geno,test_cindex,
                        val_total_loss_path_geno, val_total_loss_path, val_total_loss_geno,
                        test_total_loss_path_geno, test_total_loss_path, test_total_loss_geno]
        '''
        if args.modality == "BezierSurv":
       
            results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss),result_save_fold = _step0731(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader,test_loader)
            return  results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss),result_save_fold
        else:
            results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss),result_save_fold = _step0731(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader,test_loader)
        
    else:
        results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss),result_save_fold= _step0731(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)
        return  results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss),result_save_fold
    return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss),result_save_fold


  
def load_model_from_dict(model, save_path):
    try:
        # 加载保存的模型参数字典
        checkpoint = torch.load(save_path)
        # 获取模型中当前的参数字典
        model_dict = model.state_dict()

        # 过滤掉不需要的键
        # 这里假设 checkpoint 中保存了完整的模型信息，包括不匹配的部分
        filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}

        # 更新模型中的参数
        model_dict.update(filtered_checkpoint)
        model.load_state_dict(model_dict)

        print("[Check] Model reload success from {}".format(save_path))
        
        # 返回更新后的模型
        return model
    except Exception as e:
        print("[Error] Reloading failed: {}".format(str(e)))
        # 在加载失败时返回 None
        return None




