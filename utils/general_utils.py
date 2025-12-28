#----> general imports
import torch
import numpy as np
import torch.nn as nn
import pdb
import os
import pandas as pd 

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _prepare_for_experiment(args):
    r"""
    Creates experiment code which will be used for identifying the experiment later on. Uses the experiment code to make results dir.
    Prints and logs the important settings of the experiment. Loads the pathway composition dataframe and stores in args for future use.

    Args:
        - args : argparse.Namespace
    
    Returns:
        - args : argparse.Namespace

    """

    args.device = device
    print(args.device)
    args.split_dir = os.path.join("splits", args.which_splits, args.study)
    args.combined_study = args.study
    args = _get_custom_exp_code(args)
    _seed_torch(args.seed)

    assert os.path.isdir(args.split_dir)
    print('Split dir:', args.split_dir)

    #---> where to stroe the experiment related assets
    _create_results_dir(args)

    #---> store the settings
    settings = {'num_splits': args.k, 
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs, 
                'results_dir': args.results_dir, 
                'lr': args.lr,
                'experiment': args.study,
                'reg': args.reg,
                # 'label_frac': args.label_frac,
                'bag_loss': args.bag_loss,
                'seed': args.seed,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt,
                "num_patches":args.num_patches,
                "dropout":args.encoder_dropout,
                "type_of_path":args.type_of_path,
                'split_dir': args.split_dir
                }
    
    #---> bookkeping
    _print_and_log_experiment(args, settings)

    #---> load composition df 
    composition_df = pd.read_csv("./datasets_csv/pathway_compositions/{}_comps.csv".format(args.type_of_path), index_col=0)
    composition_df.sort_index(inplace=True)
    args.composition_df = composition_df

    return args

def _print_and_log_experiment(args, settings):
    r"""
    Prints the expeirmental settings and stores them in a file 
    
    Args:
        - args : argspace.Namespace
        - settings : dict 
    
    Return:
        - None
        
    """
    with open(args.results_dir + '/experiment_{}.txt'.format(args.param_code), 'w') as f:
        print(settings, file=f)

    f.close()

    print("")
    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))
    print("")

def _get_custom_exp_code(args):
    r"""
    Updates the argparse.NameSpace with a custom experiment code.

    Args:
        - args (NameSpace)

    Returns:
        - args (NameSpace)

    """
    dataset_path = 'datasets_csv/all_survival_endpoints'
    param_code = ''

    #----> Study 
    param_code += args.study + "_"
    param_code += '_nclass%s' % args.n_classes
    param_code += '_sp' if args.use_self_paced else '_nosp'

    #----> Loss Function
    # param_code += '_%s' % args.bag_loss
    # param_code += '_a%s' % str(args.alpha_surv)
    
    #----> Learning Rate
    # param_code += '_lr%s' % format(args.lr, '.0e')

    #----> Regularization
    # if args.reg_type == 'L1':
    #   param_code += '_%sreg%s' % (args.reg_type, format(args.reg, '.0e'))

    # if args.reg and args.reg_type == "L2":
    param_code += "_l2Weight_{}".format(args.reg)

    param_code += '_%s' % args.which_splits.split("_")[0]

    #----> Batch Size
    param_code += '_b%s' % str(args.batch_size)

    # label col 
    param_code += "_" + args.label_col

    param_code += "_dim1_" + str(args.encoding_dim)
    # param_code += "_dim2_" + str(args.encoding_layer_2_dim)
    
    param_code += "_patches_" + str(args.num_patches)
    # param_code += "_dropout_" + str(args.encoder_dropout)

    param_code += "_wsiDim_" + str(args.wsi_projection_dim)
    param_code += "_epochs_" + str(args.max_epochs)
    param_code += "_fusion_" + str(args.fusion)
    param_code += "_modality_" + str(args.modality)
    param_code += "_pathT_" + str(args.type_of_path)
    param_code += "_warmup_" + str(args.warmup_epochs)
    param_code += "_ctrlp_" + str(args.n_ctrl_points)
    param_code += "_seed_" + str(args.seed)
    param_code += "_mil_" + str(args.mil_model_type)
    
    # 用短标志串拼接
    flags = []
    if args.use_ib_path: flags.append("IBP")
    if args.use_ib_geno: flags.append("IBG")
    if args.use_ib_fusion: flags.append("IBF")
    if args.use_align_loss: flags.append("Align")
    if args.use_sim_loss: flags.append("Sim")
    if args.use_kl_align: flags.append("KL")
    if args.use_self_paced_ablation: flags.append("NoSP")
    if args.spl_on_censored_only: flags.append("spl_c")

    if flags:
        param_code += "_" + "_".join(flags)
    #----> Updating
    args.param_code = param_code
    args.dataset_path = dataset_path

    return args


def _seed_torch(seed=7):
    r"""
    Sets custom seed for torch 

    Args:
        - seed : Int 
    
    Returns:
        - None

    """
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _create_results_dir(args):
    r"""
    Creates a dir to store results for this experiment. Adds .gitignore 
    
    Args:
        - args: argspace.Namespace
    
    Return:
        - None 
    
    """
    if args.prototype_mode !='bezier_gmm':
        args.results_dir = os.path.join("./results_20250806", f"{args.results_dir}_{args.prototype_mode}")

    else:
        args.results_dir = os.path.join("./results_20250806", args.results_dir) # create an experiment specific subdir in the results dir 
    print(f"Results will be saved in: {args.results_dir}")
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
        #---> add gitignore to results dir
        f = open(os.path.join(args.results_dir, ".gitignore"), "w")
        f.write("*\n")
        f.write("*/\n")
        f.write("!.gitignore")
        f.close()
    
    #---> results for this specific experiment
    args.results_dir = os.path.join(args.results_dir, args.param_code)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

def _get_start_end(args):
    r"""
    Which folds are we training on
    
    Args:
        - args : argspace.Namespace
    
    Return:
       folds : np.array 
    
    """
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    folds = np.arange(start, end)
    return folds


def merge_stats_rows(ds_list, ds_names=None, combined_name="Combined"):
    """
    ds_list: [dataset_obj_train, dataset_obj_test, ...]
    ds_names: ["Train", "Test"]  # 可选，用于单行 LaTeX 标注
    返回: (rows, combined_row)
      rows: 每个数据集各自的 LaTeX 行（若 ds_names 提供，则用之）
      combined_row: 合并后的 LaTeX 行
    """
    # 先各自算
    per_stats = [ds.compute_stats() for ds in ds_list]

    # 逐项合并
    total_patients = sum(s['patients'] for s in per_stats)
    total_wsis     = sum(s['wsis']     for s in per_stats)
    total_patches  = sum(s['patches']  for s in per_stats)

    # 非删失占比 = (总非删失人数 / 总人数)
    # 各 split 的 non_censored_ratio 是比例，这里要按人数加权还原再合并
    total_noncensored = sum(int(s['patients'] * s['non_censored_ratio']) for s in per_stats)
    # 若需要精确（避免取整误差），用浮点：
    total_noncensored = sum((s['patients'] * s['non_censored_ratio']) for s in per_stats)

    combined_non_cens_ratio = (total_noncensored / total_patients) if total_patients > 0 else 0.0
    combined_patches_per_wsi = (total_patches / total_wsis) if total_wsis > 0 else 0.0

    # 生成各自行
    rows = []
    if ds_names is None:
        rows = [ds.to_latex_row() for ds in ds_list]
    else:
        rows = [ds.to_latex_row(name) for ds, name in zip(ds_list, ds_names)]

    # 生成合并行
    combined_row = (f"{combined_name} & "
                    f"{combined_non_cens_ratio*100:.1f}\\% & "
                    f"{int(total_patients)} & "
                    f"{int(total_wsis)} & "
                    f"{int(total_patches):,} & "
                    f"{combined_patches_per_wsi:.1f} \\\\")

    return rows, combined_row




def _save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].metadata['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val'])

    df.to_csv(filename)
    print()

def _series_intersection(s1, s2):
    r"""
    Return insersection of two sets
    
    Args:
        - s1 : set
        - s2 : set 
    
    Returns:
        - pd.Series
    
    """
    return pd.Series(list(set(s1) & set(s2)))

def _print_network(results_dir, net):
    r"""

    Print the model in terminal and also to a text file for storage 
    
    Args:
        - results_dir : String 
        - net : PyTorch model 
    
    Returns:
        - None 
    
    """
    num_params = 0
    num_params_train = 0

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

    # print(net)

    fname = "model_" + results_dir.split("/")[-1] + ".txt"
    path = os.path.join(results_dir, fname)
    f = open(path, "w")
    f.write(str(net))
    f.write("\n")
    f.write('Total number of parameters: %d \n' % num_params)
    f.write('Total number of trainable parameters: %d \n' % num_params_train)
    f.close()


def _collate_omics(batch):
    r"""
    Collate function for the unimodal omics models 
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omics : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        
    """
  
    img = torch.ones([1,1])
    omics = torch.stack([item[1] for item in batch], dim = 0)
    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    return [img, omics, label, event_time, c, clinical_data_list]


def _collate_wsi_omics(batch):
    r"""
    Collate function for the unimodal wsi and multimodal wsi + omics  models 
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omics : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        - mask : torch.Tensor
        
    """
  
    img = torch.stack([item[0] for item in batch])
    omics = torch.stack([item[1] for item in batch], dim = 0)
    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    mask = torch.stack([item[6] for item in batch], dim=0)

    return [img, omics, label, event_time, c, clinical_data_list, mask]

def _collate_MCAT(batch):
    r"""
    Collate function MCAT (pathways version) model
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omic1 : torch.Tensor 
        - omic2 : torch.Tensor 
        - omic3 : torch.Tensor 
        - omic4 : torch.Tensor 
        - omic5 : torch.Tensor 
        - omic6 : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        
    """
    
    img = torch.stack([item[0] for item in batch])

    omic1 = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    omic2 = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    omic3 = torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    omic4 = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
    omic5 = torch.cat([item[5] for item in batch], dim = 0).type(torch.FloatTensor)
    omic6 = torch.cat([item[6] for item in batch], dim = 0).type(torch.FloatTensor)


    label = torch.LongTensor([item[7].long() for item in batch])
    event_time = torch.FloatTensor([item[8] for item in batch])
    c = torch.FloatTensor([item[9] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[10])

    mask = torch.stack([item[11] for item in batch], dim=0)

    return [img, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, clinical_data_list, mask]

def _collate_ProSurv(batch):
    r"""
    Collate function for the unimodal wsi and multimodal wsi + omics  models 
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omics : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        - mask : torch.Tensor
        
    """
  
    img = torch.stack([item[0] for item in batch])
    omics = torch.stack([item[1].unsqueeze(0) for item in batch], dim = 0)
    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    mask = torch.stack([item[6] for item in batch], dim=0)
    case_list = [item[7] for item in batch]
    return [img, omics, label, event_time, c, clinical_data_list, mask,case_list]


def _collate_MCAT_batch(batch):
    r"""
    Collate function for MCAT (pathways version) model

    Args:
        - batch

    Returns:
        - img : torch.Tensor
        - omic1~omic6 : torch.Tensor
        - label : torch.LongTensor
        - event_time : torch.FloatTensor
        - c : torch.FloatTensor
        - clinical_data_list : List
        - mask : torch.Tensor
    """

    img = torch.stack([item[0] for item in batch], dim=0)

    omic1 = torch.stack([item[1] for item in batch], dim=0).type(torch.FloatTensor)
    omic2 = torch.stack([item[2] for item in batch], dim=0).type(torch.FloatTensor)
    omic3 = torch.stack([item[3] for item in batch], dim=0).type(torch.FloatTensor)
    omic4 = torch.stack([item[4] for item in batch], dim=0).type(torch.FloatTensor)
    omic5 = torch.stack([item[5] for item in batch], dim=0).type(torch.FloatTensor)
    omic6 = torch.stack([item[6] for item in batch], dim=0).type(torch.FloatTensor)

    label = torch.LongTensor([item[7].long() for item in batch])
    event_time = torch.FloatTensor([item[8] for item in batch])
    c = torch.FloatTensor([item[9] for item in batch])

    clinical_data_list = [item[10] for item in batch]

    mask = torch.stack([item[11] for item in batch], dim=0)
    case_list = [item[12] for item in batch]
    return [img, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, clinical_data_list, mask,case_list]


def _collate_survpath(batch):
    r"""
    Collate function for survpath
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omic_data_list : List
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        - mask : torch.Tensor
        
    """
    
    img = torch.stack([item[0] for item in batch])

    omic_data_list = []
    for item in batch:
        omic_data_list.append(item[1])

    label = torch.LongTensor([item[2].long() for item in batch])
    event_time = torch.FloatTensor([item[3] for item in batch])
    c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    mask = torch.stack([item[6] for item in batch], dim=0)

    return [img, omic_data_list, label, event_time, c, clinical_data_list, mask]

def _make_weights_for_balanced_classes_split(dataset):
    r"""
    Returns the weights for each class. The class will be sampled proportionally.
    
    Args: 
        - dataset : SurvivalDataset
    
    Returns:
        - final_weights : torch.DoubleTensor 
    
    """
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                   
        weight[idx] = weight_per_class[y]   

    final_weights = torch.DoubleTensor(weight)

    return final_weights

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)


def _get_split_loader(args, split_dataset, training = False, testing = False, weighted = False, batch_size=1):
    r"""
    Take a dataset and make a dataloader from it using a custom collate function. 

    Args:
        - args : argspace.Namespace
        - split_dataset : SurvivalDataset
        - training : Boolean
        - testing : Boolean
        - weighted : Boolean 
        - batch_size : Int 
    
    Returns:
        - loader : Pytorch Dataloader 
    
    """

    kwargs = {'num_workers': 0} if device.type == "cuda" else {}
    
    if args.modality in ["omics", "snn", "mlp_per_path"]:
        collate_fn = _collate_omics
    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways","s4mil_wsi", "s4mil_wsi_pathways",
                            "deepmisl_wsi", "deepmisl_wsi_pathways","dsmil_wsi_pathways","dsmil_wsi", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
        collate_fn = _collate_wsi_omics
    elif args.modality in ["coattn", "coattn_motcat","coattn_motcat_bezier"]:  
        # collate_fn = _collate_MCAT
        # 替换一下 希望支撑batch_size 能加速
        collate_fn =_collate_MCAT_batch
    elif args.modality == "survpath":
         collate_fn = _collate_survpath 
    elif args.modality in ["ProSurv","DisSurv","BezierSurv"]:
         collate_fn = _collate_ProSurv 
    else:
        raise NotImplementedError

    if not testing:
        if training:
            if weighted:
                weights = _make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_fn, drop_last=False, **kwargs)	
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_fn, drop_last=False, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_fn, drop_last=False, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SubsetSequentialSampler(ids), collate_fn = collate_fn, drop_last=False, **kwargs )

    return loader
