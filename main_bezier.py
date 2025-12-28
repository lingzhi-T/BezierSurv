#----> pytorch imports
import torch

#----> general imports
import pandas as pd
import numpy as np
import pdb
import os
from timeit import default_timer as timer
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val_0731
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment

from utils.process_args import _process_args
seed =1
import random
import numpy as np
import csv
random.seed(seed)
np.random.seed(seed)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def save_to_csv(file_path, data, mode='a'):
    """将数据保存到CSV文件"""
    with open(file_path, mode, newline='') as file:
        writer = csv.writer(file)
        # 如果是新文件，写入表头
        if mode == 'w':
            # writer.writerow(['fold', 'epoch', 'test_cindex','test_cindex_path_geno', 'test_cindex_path', 'test_cindex_geno',
            #                  'val_cindex_path', 'val_cindex_geno', 'val_cindex_path_geno', 
            #                  'val_loss_path', 'val_loss_geno', 'val_loss_path_geno', 
            #                  'test_loss_path_geno', 'test_loss_path', 'test_loss_geno'
            #                  ])
            writer.writerow([
            'fold',
            'epoch',

            # 测试集平均指标（含 IPCW）
            'test_c_index',
            'test_c_index_ipcw',
            'test_IBS',
            # 'test_BS',
            'test_iauc',

            # 测试集详细指标，path_geno -> path -> geno，含 IPCW
            'test_c_index_path_geno',
            'test_c_index_path',
            'test_c_index_geno',

            'test_c_index_ipcw_path_geno',
            'test_c_index_ipcw_path',
            'test_c_index_ipcw_geno',

            'test_IBS_path_geno',
            'test_IBS_path',
            'test_IBS_geno',

            # 'test_BS_path_geno',
            # 'test_BS_path',
            # 'test_BS_geno',

            'test_iauc_path_geno',
            'test_iauc_path',
            'test_iauc_geno',

            # 验证集平均指标（含 IPCW）
            'val_c_index',
            'val_c_index_ipcw',
            'val_IBS',
            # 'val_BS',
            'val_iauc',

            # 验证集详细指标，path_geno -> path -> geno，含 IPCW
            'val_c_index_path_geno',
            'val_c_index_path',
            'val_c_index_geno',

            'val_c_index_ipcw_path_geno',
            'val_c_index_ipcw_path',
            'val_c_index_ipcw_geno',

            'val_IBS_path_geno',
            'val_IBS_path',
            'val_IBS_geno',

            # 'val_BS_path_geno',
            # 'val_BS_path',
            # 'val_BS_geno',

            'val_iauc_path_geno',
            'val_iauc_path',
            'val_iauc_geno',

            # 损失值
            # 'val_total_loss_path_geno',
            # 'val_total_loss_path',
            # 'val_total_loss_geno',
            # 'test_total_loss_path_geno',
            # 'test_total_loss_path',
            # 'test_total_loss_geno',
        ])

        writer.writerow(data)

def main(args):

    #----> prep for 5 fold cv study
    folds = _get_start_end(args)
    
    #----> storing the val and test cindex for 5 fold cv
    # 用于保存每一折结果的列表
    all_val_cindex_path = []
    all_val_cindex_geno = []
    all_val_cindex_path_geno = []
    all_test_cindex_path_geno = []
    all_test_cindex_path = []
    all_test_cindex_geno = []
    all_test_cindex = []

    all_val_total_loss_path_geno = []
    all_val_total_loss_path = []
    all_val_total_loss_geno = []
    all_test_total_loss_path_geno = []
    all_test_total_loss_path = []
    all_test_total_loss_geno = []
    all_val_c_index = []
    all_val_c_index_ipcw = []
    all_val_IBS = []
    all_val_BS = []
    all_val_iauc = []

    all_val_c_index_path_geno = []
    all_val_c_index_path = []
    all_val_c_index_geno = []

    all_val_c_index_ipcw_path_geno = []
    all_val_c_index_ipcw_path = []
    all_val_c_index_ipcw_geno = []

    all_val_IBS_path_geno = []
    all_val_IBS_path = []
    all_val_IBS_geno = []

    all_val_BS_path_geno = []
    all_val_BS_path = []
    all_val_BS_geno = []

    all_val_iauc_path_geno = []
    all_val_iauc_path = []
    all_val_iauc_geno = []

    all_test_c_index = []
    all_test_c_index_ipcw = []
    all_test_IBS = []
    all_test_BS = []
    all_test_iauc = []

    all_test_c_index_path_geno = []
    all_test_c_index_path = []
    all_test_c_index_geno = []

    all_test_c_index_ipcw_path_geno = []
    all_test_c_index_ipcw_path = []
    all_test_c_index_ipcw_geno = []

    all_test_IBS_path_geno = []
    all_test_IBS_path = []
    all_test_IBS_geno = []

    all_test_BS_path_geno = []
    all_test_BS_path = []
    all_test_BS_geno = []

    all_test_iauc_path_geno = []
    all_test_iauc_path = []
    all_test_iauc_geno = []
    for i in folds:
        # if i == 4:
            setup_seed(args.seed)
            datasets = args.dataset_factory.return_splits(
                args,
                csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
                fold=i
            )
            
            print("Created train and val datasets for fold {}".format(i))

            results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss),result_save_fold = _train_val_0731(datasets, i, args)

            '''
            data_save = [
                        epoch,

                        # 测试集平均指标（含 IPCW）
                        test_c_index,
                        test_c_index_ipcw,
                        test_IBS,
                        test_BS,
                        test_iauc,

                        # 测试集详细指标，按 path_geno -> path -> geno 顺序，含 IPCW
                        test_c_index_path_geno,
                        test_c_index_path,
                        test_c_index_geno,

                        test_c_index_ipcw_path_geno,
                        test_c_index_ipcw_path,
                        test_c_index_ipcw_geno,

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
                        val_c_index,
                        val_c_index_ipcw,
                        val_IBS,
                        val_BS,
                        val_iauc,

                        # 验证集详细指标，按 path_geno -> path -> geno 顺序，含 IPCW
                        val_c_index_path_geno,
                        val_c_index_path,
                        val_c_index_geno,

                        val_c_index_ipcw_path_geno,
                        val_c_index_ipcw_path,
                        val_c_index_ipcw_geno,

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

            '''
            # epoch = result_save_fold[0]
            # val_cindex_path = result_save_fold[1]  # 根据 result_save_fold 提取对应的 C-index 和 loss
            # val_cindex_geno = result_save_fold[2]
            # val_cindex_path_geno = result_save_fold[3]
            # val_cindex = result_save_fold[4]
            
            # test_cindex_path_geno = result_save_fold[5]
            # test_cindex_path = result_save_fold[6]
            # test_cindex_geno = result_save_fold[7]
            # test_cindex = result_save_fold[8]
            
            # val_total_loss_path_geno = result_save_fold[9]
            # val_total_loss_path = result_save_fold[10]
            # val_total_loss_geno = result_save_fold[11]
            
            # test_total_loss_path_geno = result_save_fold[12]
            # test_total_loss_path = result_save_fold[13]
            # test_total_loss_geno = result_save_fold[14]
            epoch = result_save_fold[0]

            # 测试集平均指标
            test_c_index = result_save_fold[1]
            test_c_index_ipcw = result_save_fold[2]
            test_IBS = result_save_fold[3]
            # test_BS = result_save_fold[4]
            test_iauc = result_save_fold[5]

            # 测试集详细指标
            test_c_index_path_geno = result_save_fold[6]
            test_c_index_path = result_save_fold[7]
            test_c_index_geno = result_save_fold[8]

            test_c_index_ipcw_path_geno = result_save_fold[9]
            test_c_index_ipcw_path = result_save_fold[10]
            test_c_index_ipcw_geno = result_save_fold[11]

            test_IBS_path_geno = result_save_fold[12]
            test_IBS_path = result_save_fold[13]
            test_IBS_geno = result_save_fold[14]

            # test_BS_path_geno = result_save_fold[15]
            # test_BS_path = result_save_fold[16]
            # test_BS_geno = result_save_fold[17]

            test_iauc_path_geno = result_save_fold[18]
            test_iauc_path = result_save_fold[19]
            test_iauc_geno = result_save_fold[20]

            # 验证集平均指标
            val_c_index = result_save_fold[21]
            val_c_index_ipcw = result_save_fold[22]
            val_IBS = result_save_fold[23]
            # val_BS = result_save_fold[24]
            val_iauc = result_save_fold[25]

            # 验证集详细指标
            val_c_index_path_geno = result_save_fold[26]
            val_c_index_path = result_save_fold[27]
            val_c_index_geno = result_save_fold[28]

            val_c_index_ipcw_path_geno = result_save_fold[29]
            val_c_index_ipcw_path = result_save_fold[30]
            val_c_index_ipcw_geno = result_save_fold[31]

            val_IBS_path_geno = result_save_fold[32]
            val_IBS_path = result_save_fold[33]
            val_IBS_geno = result_save_fold[34]

            # val_BS_path_geno = result_save_fold[35]
            # val_BS_path = result_save_fold[36]
            # val_BS_geno = result_save_fold[37]

            val_iauc_path_geno = result_save_fold[38]
            val_iauc_path = result_save_fold[39]
            val_iauc_geno = result_save_fold[40]

            # 损失值
            val_total_loss_path_geno = result_save_fold[41]
            val_total_loss_path = result_save_fold[42]
            val_total_loss_geno = result_save_fold[43]

            test_total_loss_path_geno = result_save_fold[44]
            test_total_loss_path = result_save_fold[45]
            test_total_loss_geno = result_save_fold[46]

         # 将每个折的结果添加到列表中
            # all_val_cindex_path.append(val_cindex_path)
            # all_val_cindex_geno.append(val_cindex_geno)
            # all_val_cindex_path_geno.append(val_cindex_path_geno)
            # all_test_cindex_path_geno.append(test_cindex_path_geno)
            # all_test_cindex_path.append(test_cindex_path)
            # all_test_cindex_geno.append(test_cindex_geno)
            # all_test_cindex.append(test_cindex)
            all_val_c_index.append(val_c_index)
            all_val_c_index_ipcw.append(val_c_index_ipcw)
            all_val_IBS.append(val_IBS)
            # all_val_BS.append(val_BS)
            all_val_iauc.append(val_iauc)

            all_val_c_index_path_geno.append(val_c_index_path_geno)
            all_val_c_index_path.append(val_c_index_path)
            all_val_c_index_geno.append(val_c_index_geno)

            all_val_c_index_ipcw_path_geno.append(val_c_index_ipcw_path_geno)
            all_val_c_index_ipcw_path.append(val_c_index_ipcw_path)
            all_val_c_index_ipcw_geno.append(val_c_index_ipcw_geno)

            all_val_IBS_path_geno.append(val_IBS_path_geno)
            all_val_IBS_path.append(val_IBS_path)
            all_val_IBS_geno.append(val_IBS_geno)

            # all_val_BS_path_geno.append(val_BS_path_geno)
            # all_val_BS_path.append(val_BS_path)
            # all_val_BS_geno.append(val_BS_geno)

            all_val_iauc_path_geno.append(val_iauc_path_geno)
            all_val_iauc_path.append(val_iauc_path)
            all_val_iauc_geno.append(val_iauc_geno)

            all_test_c_index.append(test_c_index)
            all_test_c_index_ipcw.append(test_c_index_ipcw)
            all_test_IBS.append(test_IBS)
            # all_test_BS.append(test_BS)
            all_test_iauc.append(test_iauc)

            all_test_c_index_path_geno.append(test_c_index_path_geno)
            all_test_c_index_path.append(test_c_index_path)
            all_test_c_index_geno.append(test_c_index_geno)

            all_test_c_index_ipcw_path_geno.append(test_c_index_ipcw_path_geno)
            all_test_c_index_ipcw_path.append(test_c_index_ipcw_path)
            all_test_c_index_ipcw_geno.append(test_c_index_ipcw_geno)

            all_test_IBS_path_geno.append(test_IBS_path_geno)
            all_test_IBS_path.append(test_IBS_path)
            all_test_IBS_geno.append(test_IBS_geno)

            # all_test_BS_path_geno.append(test_BS_path_geno)
            # all_test_BS_path.append(test_BS_path)
            # all_test_BS_geno.append(test_BS_geno)

            all_test_iauc_path_geno.append(test_iauc_path_geno)
            all_test_iauc_path.append(test_iauc_path)
            all_test_iauc_geno.append(test_iauc_geno)

            all_val_total_loss_path_geno.append(val_total_loss_path_geno)
            all_val_total_loss_path.append(val_total_loss_path)
            all_val_total_loss_geno.append(val_total_loss_geno)
            all_test_total_loss_path_geno.append(test_total_loss_path_geno)
            all_test_total_loss_path.append(test_total_loss_path)
            all_test_total_loss_geno.append(test_total_loss_geno)
        
            #write results to pkl
            filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
            print("Saving results...")
            _save_pkl(filename, results)
            # === 打印每个折的结果 ===
            print(f"Fold {i}:")
            print(f"Val C-index: {val_cindex}, Test C-index: {test_c_index}")
            print(f"Val Loss: {val_total_loss_path_geno}, Test Loss: {test_total_loss_path_geno}")
            # === 将每折的结果存储到CSV ===

            data = [
                i,  # fold
                epoch,

                # 测试集平均指标（含 IPCW）
                test_c_index,
                test_c_index_ipcw,
                test_IBS,
                # test_BS,
                test_iauc,

                # 测试集详细指标，按 path_geno -> path -> geno 顺序，含 IPCW
                test_c_index_path_geno,
                test_c_index_path,
                test_c_index_geno,

                test_c_index_ipcw_path_geno,
                test_c_index_ipcw_path,
                test_c_index_ipcw_geno,

                test_IBS_path_geno,
                test_IBS_path,
                test_IBS_geno,

                # test_BS_path_geno,
                # test_BS_path,
                # test_BS_geno,

                test_iauc_path_geno,
                test_iauc_path,
                test_iauc_geno,

                # 验证集平均指标（含 IPCW）
                val_c_index,
                val_c_index_ipcw,
                val_IBS,
                # val_BS,
                val_iauc,

                # 验证集详细指标，按 path_geno -> path -> geno 顺序，含 IPCW
                val_c_index_path_geno,
                val_c_index_path,
                val_c_index_geno,

                val_c_index_ipcw_path_geno,
                val_c_index_ipcw_path,
                val_c_index_ipcw_geno,

                val_IBS_path_geno,
                val_IBS_path,
                val_IBS_geno,

                # val_BS_path_geno,
                # val_BS_path,
                # val_BS_geno,

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

            # 如果是第一次保存，创建文件并写入表头；否则追加数据
            save_to_csv(os.path.join(args.results_dir, "result.csv"), data, mode='w' if i == 0 else 'a')

            # 计算每个指标的平均值

    # 计算均值
    final_val_c_index = np.mean(all_val_c_index)
    final_val_c_index_ipcw = np.mean(all_val_c_index_ipcw)
    final_val_IBS = np.mean(all_val_IBS)
    # final_val_BS = np.mean(all_val_BS)
    final_val_iauc = np.mean(all_val_iauc)

    final_val_c_index_path_geno = np.mean(all_val_c_index_path_geno)
    final_val_c_index_path = np.mean(all_val_c_index_path)
    final_val_c_index_geno = np.mean(all_val_c_index_geno)

    final_val_c_index_ipcw_path_geno = np.mean(all_val_c_index_ipcw_path_geno)
    final_val_c_index_ipcw_path = np.mean(all_val_c_index_ipcw_path)
    final_val_c_index_ipcw_geno = np.mean(all_val_c_index_ipcw_geno)

    final_val_IBS_path_geno = np.mean(all_val_IBS_path_geno)
    final_val_IBS_path = np.mean(all_val_IBS_path)
    final_val_IBS_geno = np.mean(all_val_IBS_geno)

    # final_val_BS_path_geno = np.mean(all_val_BS_path_geno)
    # final_val_BS_path = np.mean(all_val_BS_path)
    # final_val_BS_geno = np.mean(all_val_BS_geno)

    final_val_iauc_path_geno = np.mean(all_val_iauc_path_geno)
    final_val_iauc_path = np.mean(all_val_iauc_path)
    final_val_iauc_geno = np.mean(all_val_iauc_geno)


    final_test_c_index = np.mean(all_test_c_index)
    final_test_c_index_ipcw = np.mean(all_test_c_index_ipcw)
    final_test_IBS = np.mean(all_test_IBS)
    # final_test_BS = np.mean(all_test_BS)
    final_test_iauc = np.mean(all_test_iauc)

    final_test_c_index_path_geno = np.mean(all_test_c_index_path_geno)
    final_test_c_index_path = np.mean(all_test_c_index_path)
    final_test_c_index_geno = np.mean(all_test_c_index_geno)

    final_test_c_index_ipcw_path_geno = np.mean(all_test_c_index_ipcw_path_geno)
    final_test_c_index_ipcw_path = np.mean(all_test_c_index_ipcw_path)
    final_test_c_index_ipcw_geno = np.mean(all_test_c_index_ipcw_geno)

    final_test_IBS_path_geno = np.mean(all_test_IBS_path_geno)
    final_test_IBS_path = np.mean(all_test_IBS_path)
    final_test_IBS_geno = np.mean(all_test_IBS_geno)

    # final_test_BS_path_geno = np.mean(all_test_BS_path_geno)
    # final_test_BS_path = np.mean(all_test_BS_path)
    # final_test_BS_geno = np.mean(all_test_BS_geno)

    final_test_iauc_path_geno = np.mean(all_test_iauc_path_geno)
    final_test_iauc_path = np.mean(all_test_iauc_path)
    final_test_iauc_geno = np.mean(all_test_iauc_geno)


    # final_val_total_loss_path_geno = np.mean(all_val_total_loss_path_geno)
    # final_val_total_loss_path = np.mean(all_val_total_loss_path)
    # final_val_total_loss_geno = np.mean(all_val_total_loss_geno)
    # final_test_total_loss_path_geno = np.mean(all_test_total_loss_path_geno)
    # final_test_total_loss_path = np.mean(all_test_total_loss_path)
    # final_test_total_loss_geno = np.mean(all_test_total_loss_geno)


    # 计算方差
    var_val_c_index = np.var(all_val_c_index)
    var_val_c_index_ipcw = np.var(all_val_c_index_ipcw)
    var_val_IBS = np.var(all_val_IBS)
    # var_val_BS = np.var(all_val_BS)
    var_val_iauc = np.var(all_val_iauc)

    var_val_c_index_path_geno = np.var(all_val_c_index_path_geno)
    var_val_c_index_path = np.var(all_val_c_index_path)
    var_val_c_index_geno = np.var(all_val_c_index_geno)

    var_val_c_index_ipcw_path_geno = np.var(all_val_c_index_ipcw_path_geno)
    var_val_c_index_ipcw_path = np.var(all_val_c_index_ipcw_path)
    var_val_c_index_ipcw_geno = np.var(all_val_c_index_ipcw_geno)

    var_val_IBS_path_geno = np.var(all_val_IBS_path_geno)
    var_val_IBS_path = np.var(all_val_IBS_path)
    var_val_IBS_geno = np.var(all_val_IBS_geno)

    # var_val_BS_path_geno = np.var(all_val_BS_path_geno)
    # var_val_BS_path = np.var(all_val_BS_path)
    # var_val_BS_geno = np.var(all_val_BS_geno)

    var_val_iauc_path_geno = np.var(all_val_iauc_path_geno)
    var_val_iauc_path = np.var(all_val_iauc_path)
    var_val_iauc_geno = np.var(all_val_iauc_geno)


    var_test_c_index = np.var(all_test_c_index)
    var_test_c_index_ipcw = np.var(all_test_c_index_ipcw)
    var_test_IBS = np.var(all_test_IBS)
    # var_test_BS = np.var(all_test_BS)
    var_test_iauc = np.var(all_test_iauc)

    var_test_c_index_path_geno = np.var(all_test_c_index_path_geno)
    var_test_c_index_path = np.var(all_test_c_index_path)
    var_test_c_index_geno = np.var(all_test_c_index_geno)

    var_test_c_index_ipcw_path_geno = np.var(all_test_c_index_ipcw_path_geno)
    var_test_c_index_ipcw_path = np.var(all_test_c_index_ipcw_path)
    var_test_c_index_ipcw_geno = np.var(all_test_c_index_ipcw_geno)

    var_test_IBS_path_geno = np.var(all_test_IBS_path_geno)
    var_test_IBS_path = np.var(all_test_IBS_path)
    var_test_IBS_geno = np.var(all_test_IBS_geno)

    # var_test_BS_path_geno = np.var(all_test_BS_path_geno)
    # var_test_BS_path = np.var(all_test_BS_path)
    # var_test_BS_geno = np.var(all_test_BS_geno)

    var_test_iauc_path_geno = np.var(all_test_iauc_path_geno)
    var_test_iauc_path = np.var(all_test_iauc_path)
    var_test_iauc_geno = np.var(all_test_iauc_geno)


    # 组装均值数据
    data_avg = [
        None, None,
        
        final_test_c_index,
        final_test_c_index_ipcw,
        final_test_IBS,
        # final_test_BS,
        final_test_iauc,

        final_test_c_index_path_geno,
        final_test_c_index_path,
        final_test_c_index_geno,

        final_test_c_index_ipcw_path_geno,
        final_test_c_index_ipcw_path,
        final_test_c_index_ipcw_geno,

        final_test_IBS_path_geno,
        final_test_IBS_path,
        final_test_IBS_geno,

        # final_test_BS_path_geno,
        # final_test_BS_path,
        # final_test_BS_geno,

        final_test_iauc_path_geno,
        final_test_iauc_path,
        final_test_iauc_geno,

        final_val_c_index,
        final_val_c_index_ipcw,
        final_val_IBS,
        # final_val_BS,
        final_val_iauc,

        final_val_c_index_path_geno,
        final_val_c_index_path,
        final_val_c_index_geno,

        final_val_c_index_ipcw_path_geno,
        final_val_c_index_ipcw_path,
        final_val_c_index_ipcw_geno,

        final_val_IBS_path_geno,
        final_val_IBS_path,
        final_val_IBS_geno,

        # final_val_BS_path_geno,
        # final_val_BS_path,
        # final_val_BS_geno,

        final_val_iauc_path_geno,
        final_val_iauc_path,
        final_val_iauc_geno,

        # final_val_total_loss_path_geno,
        # final_val_total_loss_path,
        # final_val_total_loss_geno,
        # final_test_total_loss_path_geno,
        # final_test_total_loss_path,
        # final_test_total_loss_geno,
    ]

    # 组装方差数据
    data_var = [
        None, None,
        
        var_test_c_index,
        var_test_c_index_ipcw,
        var_test_IBS,
        # var_test_BS,
        var_test_iauc,

        var_test_c_index_path_geno,
        var_test_c_index_path,
        var_test_c_index_geno,

        var_test_c_index_ipcw_path_geno,
        var_test_c_index_ipcw_path,
        var_test_c_index_ipcw_geno,

        var_test_IBS_path_geno,
        var_test_IBS_path,
        var_test_IBS_geno,

        # var_test_BS_path_geno,
        # var_test_BS_path,
        # var_test_BS_geno,

        var_test_iauc_path_geno,
        var_test_iauc_path,
        var_test_iauc_geno,

        var_val_c_index,
        var_val_c_index_ipcw,
        var_val_IBS,
        # var_val_BS,
        var_val_iauc,

        var_val_c_index_path_geno,
        var_val_c_index_path,
        var_val_c_index_geno,

        var_val_c_index_ipcw_path_geno,
        var_val_c_index_ipcw_path,
        var_val_c_index_ipcw_geno,

        var_val_IBS_path_geno,
        var_val_IBS_path,
        var_val_IBS_geno,

        # var_val_BS_path_geno,
        # var_val_BS_path,
        # var_val_BS_geno,

        var_val_iauc_path_geno,
        var_val_iauc_path,
        var_val_iauc_geno,

        # var_val_total_loss_path_geno,
        # var_val_total_loss_path,
        # var_val_total_loss_geno,
        # var_test_total_loss_path_geno,
        # var_test_total_loss_path,
        # var_test_total_loss_geno,
    ]

    # final_val_cindex_path = sum(all_val_cindex_path) / len(all_val_cindex_path)
    # final_val_cindex_geno = sum(all_val_cindex_geno) / len(all_val_cindex_geno)
    # final_val_cindex_path_geno = sum(all_val_cindex_path_geno) / len(all_val_cindex_path_geno)
    # final_test_cindex_path_geno = sum(all_test_cindex_path_geno) / len(all_test_cindex_path_geno)
    # final_test_cindex_path = sum(all_test_cindex_path) / len(all_test_cindex_path)
    # final_test_cindex_geno = sum(all_test_cindex_geno) / len(all_test_cindex_geno)
    # final_test_cindex = sum(all_test_cindex) / len(all_test_cindex)

    # final_val_total_loss_path_geno = sum(all_val_total_loss_path_geno) / len(all_val_total_loss_path_geno)
    # final_val_total_loss_path = sum(all_val_total_loss_path) / len(all_val_total_loss_path)
    # final_val_total_loss_geno = sum(all_val_total_loss_geno) / len(all_val_total_loss_geno)
    # final_test_total_loss_path_geno = sum(all_test_total_loss_path_geno) / len(all_test_total_loss_path_geno)
    # final_test_total_loss_path = sum(all_test_total_loss_path) / len(all_test_total_loss_path)
    # final_test_total_loss_geno = sum(all_test_total_loss_geno) / len(all_test_total_loss_geno)
    
    # data = [None, None, final_test_cindex,final_test_cindex_path_geno, final_test_cindex_path, final_test_cindex_geno, 
    #      final_val_cindex_path, final_val_cindex_geno, final_val_cindex_path_geno,
    #     final_val_total_loss_path_geno, final_val_total_loss_path, final_val_total_loss_geno,
    #     final_test_total_loss_path_geno, final_test_total_loss_path, final_test_total_loss_geno,
    #     ]  # 添加默认的 'avg' 字段

    # 如果是第一次保存，创建文件并写入表头；否则追加数据
    save_to_csv(os.path.join(args.results_dir, "result.csv"), data_avg, mode='w' if i == 0 else 'a')
    # 计算方差
    # var_val_cindex_path = np.var(all_val_cindex_path)
    # var_val_cindex_geno = np.var(all_val_cindex_geno)
    # var_val_cindex_path_geno = np.var(all_val_cindex_path_geno)
    # var_test_cindex_path_geno = np.var(all_test_cindex_path_geno)
    # var_test_cindex_path = np.var(all_test_cindex_path)
    # var_test_cindex_geno = np.var(all_test_cindex_geno)
    # var_test_cindex = np.var(all_test_cindex)

    # var_val_total_loss_path_geno = np.var(all_val_total_loss_path_geno)
    # var_val_total_loss_path = np.var(all_val_total_loss_path)
    # var_val_total_loss_geno = np.var(all_val_total_loss_geno)
    # var_test_total_loss_path_geno = np.var(all_test_total_loss_path_geno)
    # var_test_total_loss_path = np.var(all_test_total_loss_path)
    # var_test_total_loss_geno = np.var(all_test_total_loss_geno)

    # data_var = [
    #     None, None,var_test_cindex,  var_test_cindex_path_geno, var_test_cindex_path, var_test_cindex_geno, 
    #     var_val_cindex_path, var_val_cindex_geno, var_val_cindex_path_geno,
      
    #     var_val_total_loss_path_geno, var_val_total_loss_path, var_val_total_loss_geno,
    #     var_test_total_loss_path_geno, var_test_total_loss_path, var_test_total_loss_geno
    # ]  # 添加默认的 'variance' 字段

    save_to_csv(os.path.join(args.results_dir, "result.csv"), data_var, mode='a')  # 保存方差
    


if __name__ == "__main__":
    start = timer()

    #----> read the args
    args = _process_args()
    
    #----> Prep
    args = _prepare_for_experiment(args)
    
    #----> create dataset factory
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study,
        label_file=args.label_file,
        omics_dir=args.omics_dir,
        seed=args.seed, 
        print_info=True, 
        n_bins=args.n_classes, 
        label_col=args.label_col, 
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat = True if "coattn" in args.modality else False,
        is_survpath = True if args.modality == "survpath" else False,
        type_of_pathway=args.type_of_path)

    #---> perform the experiment
    results = main(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))