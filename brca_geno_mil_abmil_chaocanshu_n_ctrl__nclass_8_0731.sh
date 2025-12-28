#!/bin/bash

# 设置网格搜索的超参数列表
MIL_MODEL_TYPES=("ABMIL")   # 可选的 MIL 模型类型
# GENO_MODEL_TYPES=("SNN" "mlp")  # 可选的基因类型
GENO_MODEL_TYPES=("SNN")  # 可选的基因类型

BATCH_SIZES=(1)             # 可选的 Batch size
LR=0.0001                   # 学习率
OPT="radam"                 # 优化器
REG=0.0001                  # 正则化
ALPHA_SURV=0.5              # 生存任务的超参数
SEEDS=(1)                   # 可选的随机种子
CTRL_POINTS_LIST=(8)  # 控制点数量的列表
N_CLASSES_LIST=(8)      # 类别数量的列表

# GPU ID
for MIL_MODEL in "${MIL_MODEL_TYPES[@]}"; do
  for GENO_MODEL in "${GENO_MODEL_TYPES[@]}"; do
    for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
      for SEED in "${SEEDS[@]}"; do
        for N_CTRL_POINTS in "${CTRL_POINTS_LIST[@]}"; do  # 遍历 n_ctrl_points
          for N_CLASSES in "${N_CLASSES_LIST[@]}"; do    # 遍历 n_classes

            CUDA_VISIBLE_DEVICES=9 python main_bezier_20250729_jiasu.py \
              --study tcga_brca \
              --task survival \
              --split_dir splits \
              --which_splits 5foldcv \
              --type_of_path combine \
              --modality BezierSurv \
              --data_root_dir  /media/raid/****/svs_file/bingli_20240729/brca_processed/feat-l1-uni-B-new/pt_files \
              --label_file datasets_csv/metadata_guolv/tcga_brca.csv \
              --omics_dir datasets_csv/raw_rna_data/combine/brca \
              --results_dir results_brca-0825-new_ours_spl_on_censored_only_n_classes_${N_CLASSES}_lr_${LR}_mil_${MIL_MODEL}_geno_${GENO_MODEL}_seed_${SEED}_ctrl_points_${N_CTRL_POINTS} \
              --batch_size "${BATCH_SIZE}" \
              --lr "${LR}" \
              --opt "${OPT}" \
              --reg "${REG}" \
              --alpha_surv "${ALPHA_SURV}" \
              --max_epochs 100 \
              --encoding_dim 1024 \
              --early_stop_patience 35 \
              --label_col survival_months_dss \
              --k 5 \
              --bag_loss nll_surv \
              --n_classes "${N_CLASSES}" \
              --num_patches 4096 \
              --wsi_projection_dim 256 \
              --fusion concat \
              --warmup_epochs 50 \
              --use_self_paced \
              --spl_on_censored_only \
              --use_ib_path \
              --use_ib_geno \
              --use_align_loss \
              --use_sim_loss \
              --test_all_modalities \
              --n_ctrl_points "${N_CTRL_POINTS}" \
              --mil_model_type "${MIL_MODEL}" \
              --geno_mlp_type "${GENO_MODEL}" \
              --seed "${SEED}" \
              --use_mean_in_testing \
              --use_wandb \
              --k_start 0 \
              --k_end 1
          done
        done
      done
    done
  done
done
