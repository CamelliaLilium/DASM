DASM_ROOT="${DASM_ROOT:-$(cd "$(dirname "$0")" && pwd)}"
DASM_DATA_ROOT="${DASM_DATA_ROOT:-${DASM_ROOT}/dataset}"

# 使用SFT和PCE训练模型
# 每5个epoch评估一次域测试准确率
python model_multi.py --train_dataset combined --dataset_id QIM+PMS+LSB+AHCM_0.5_1s.pkl --steg_algorithm Transformer --use_sft --use_pce --domain_test_interval 5 --result_path ${DASM_ROOT}/training_visualization/data/models_pce
python model_multi.py --train_dataset combined --dataset_id QIM+PMS+LSB+AHCM_0.5_1s.pkl --steg_algorithm LStegT --domain_test_interval 5 --result_path ${DASM_ROOT}/training_visualization/data/models_base ；

# 提取最佳 epoch_acc 和 domain_test_acc
python -m extract_best_metrics --json ${DASM_ROOT}/models_collection/Transformer/csam_train_AHCM_LSB_PMS_QIM_to_AHCM_LSB_PMS_QIM/train_logs_QIM+PMS+LSB+AHCM_0.5_1s.json
python ${DASM_ROOT}/utils/extract_best_metrics.py --json ${DASM_ROOT}/models_collection/dasm_domain_gap/Transformer/dasm_er0.5_bs1024_rho0.03_ctau0.1_gap_seed42/train_logs_QIM+PMS+LSB+AHCM_0.5_1s.json

# 计算域差
python domain_gap_calculator.py --embedding_rates 0.1 0.3 0.5 --gpu 0

# 使用中数据集训练模型
python model_domain_generalization.py --dataset_id=QIM+PMS+LSB+AHCM_0.1_1s \
    --data_root ${DASM_DATA_ROOT}/model_train/combined_multi_mid \
    --embedding_rate=0.1 --save_model --epochs 50 --batch_size 4096 --gpu 0
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --use_contrast --contrast_tau 0.1 --rho 0.03 \
    --seed 42 --epochs 50 --batch_size 1024 --gpu 0 --save_model

cd ${DASM_ROOT}/performance
# 运行Adam
python benchmark.py --optimizer adam --batch_size 128 --epochs 5
# 运行SAM
python benchmark.py --optimizer sam --batch_size 128 --epochs 3 ;\
python benchmark.py --optimizer dasm --batch_size 128 --epochs 3 ;\
python summarize_results.py

# 域差 + DASM
python model_csam_DomainGap_v1.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.1_1s' \
    --steg_algorithm Transformer --train_domains=QIM,PMS,LSB,AHCM --test_domains=QIM,PMS,LSB,AHCM \
    --use_csam --use_domain_gap_loss \
    --gap_lambda 0.5 --gap_target_cover_pms 2.0 --gap_weight_pms 3.0 \
    --epochs 100 --domain_test_interval 5 --gpu 0 --save_model --batch_size 1024
python model_csam_DomainGap_v2.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_csam --use_adaptive_domain_gap --gap_lambda 1.0 --gap_target 2.0 --gap_temperature 0.1 \
    --epochs 50 --domain_test_interval 5 --gpu 0 --batch_size 1024 --save_model
    #--contrast_loss 0.1
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.1_1s' --embedding_rate 0.1 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --use_adaptive_domain_gap --temperature 0.5 \
    --seed 42 --epochs 100 --domain_test_interval 5 --batch_size 1024 --gpu 0 --save_model

# 超参敏感性分析
# rho
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --use_adaptive_domain_gap --contrast_tau 0.5 --gap_tau 0.5 --rho 0.05 \
    --seed 42 --epochs 100 --batch_size 1024 --gpu 1 --save_model
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --use_adaptive_domain_gap --contrast_tau 0.5 --gap_tau 0.5 --rho 0.01 \
    --seed 42 --epochs 100 --batch_size 1024 --gpu 1 --save_model
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --use_adaptive_domain_gap --contrast_tau 0.5 --gap_tau 0.5 --rho 0.03 \
    --seed 42 --epochs 100 --batch_size 1024 --gpu 1 --save_model
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --use_adaptive_domain_gap --contrast_tau 0.5 --gap_tau 0.5 --rho 0.08 \
    --seed 42 --epochs 100 --batch_size 1024 --gpu 1 --save_model
# python ${DASM_ROOT}/model_dasm_DomainGap.py \
#     --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
#     --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
#     --use_dasm --use_adaptive_domain_gap --contrast_tau 0.5 --gap_tau 0.5 --rho 0.1 \
#     --seed 42 --epochs 100 --batch_size 1024 --gpu 1 --save_model
# rho=0.03  contrast_tau
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --contrast_tau 0.1 --rho 0.03 \
    --seed 42 --epochs 100 --batch_size 1024 --gpu 0 --save_model
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --contrast_tau 0.2 --rho 0.03 \
    --seed 42 --epochs 100 --batch_size 1024 --gpu 1 --save_model
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --contrast_tau 0.8 --rho 0.03 \
    --seed 42 --epochs 100 --batch_size 1024 --gpu 1 --save_model
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --contrast_tau 1.0 --rho 0.03 \
    --seed 42 --epochs 100 --batch_size 1024 --gpu 0 --save_model
# 消融实验
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --use_contrast --contrast_tau 0.1 --rho 0.03 \
    --seed 42 --epochs 100 --batch_size 1024 --gpu 0 --save_model
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --contrast_tau 0.1 --rho 0.03 \
    --seed 42 --epochs 100 --batch_size 1024 --gpu 0 --save_model
python ${DASM_ROOT}/model_dasm_DomainGap.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --contrast_tau 0.1 --rho 0.03 \
    --seed 42 --epochs 100 --batch_size 1024 --gpu 0 --save_model


# 多域学习 SASM
python model_domain_generalization_sasm.py \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s --embedding_rate 0.5\
    --steg_algorithm Transformer --train_domains=QIM,PMS,LSB,AHCM --test_domains=QIM,PMS,LSB,AHCM \
    --use_sasm --rho 0.1 --diff_mode pairwise --mu_mode const --mu 0.1 \
    --batch_size 160 --epochs 100 --save_model --eval_step 5 --gpu 0  ;\
python model_domain_generalization_sasm.py \
  --dataset_id QIM+PMS+LSB+AHCM_0.5_1s --embedding_rate 0.5 \
  --steg_algorithm Transformer --train_domains=QIM,PMS,LSB,AHCM --test_domains=QIM,PMS,LSB,AHCM \
  --use_sasm --rho 0.05 --diff_mode pairwise --mu_mode const --mu 0.1 \
  --lambda_balance 0.2 --lambda_balance_mode const \
  --batch_size 64 --epochs 100 --save_model --eval_step 5 --gpu 0

# 多域学习 Transformer+Adam
python model_domain_generalization.py \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=QIM,PMS,LSB,AHCM \
    --data_root ${DASM_DATA_ROOT}/model_train/combined_multi \
    --test_domains=QIM,PMS,LSB,AHCM --embedding_rate=0.5 --steg_algorithm=Transformer \
    --save_model --eval_step 10 --batch_size 256 --gpu 0
# 多域学习 LStegT+Adam（已完成大数据集）
python model_domain_generalization.py \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=QIM,PMS,LSB,AHCM \
    --data_root ${DASM_DATA_ROOT}/model_train/combined_multi \
    --test_domains=QIM,PMS,LSB,AHCM --embedding_rate=0.5 --steg_algorithm=LStegT \
    --save_model --eval_step 10 --batch_size 128
# 多域学习 使用KFEF（已完成大数据集）
python model_domain_generalization.py --train_dataset=combined \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=QIM,PMS,LSB,AHCM \
    --data_root ${DASM_DATA_ROOT}/model_train/combined_multi \
    --test_domains=QIM,PMS,LSB,AHCM --embedding_rate=0.5 --steg_algorithm=KFEF \
    --save_model --eval_step 10 --batch_size 128
# 多域学习 使用FS-MDP（已完成大数据集）
python model_domain_generalization.py \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=QIM,PMS,LSB,AHCM \
    --data_root ${DASM_DATA_ROOT}/model_train/combined_multi \
    --test_domains=QIM,PMS,LSB,AHCM --embedding_rate=0.5 --steg_algorithm=FS-MDP \
    --save_model --eval_step 10 --batch_size 128
# 多域学习 DVSF+Adam（可以开始大数据集）
python model_domain_generalization.py \
  --dataset_id QIM+PMS+LSB+AHCM_0.5_1s --steg_algorithm DVSF \
  --data_root ${DASM_DATA_ROOT}/model_train/combined_multi \
  --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
  --batch_size 128 --embedding_rate 0.5 --eval_step 10 --epochs 100 --gpu 0 --save_model
# 多域学习 DAEF_VS+Adam（可以开始大数据集）
python model_domain_generalization_sam.py \
  --dataset_id QIM+PMS+LSB+AHCM_0.5_1s --steg_algorithm DAEF-VS \
  --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
  --data_root ${DASM_DATA_ROOT}/model_train/combined_multi \
  --batch_size 180 --embedding_rate 0.5 --eval_step 10 --epochs 100 --gpu 0 --save_model
# 多域学习 SFFN+Adam（可以开始大数据集）
python model_domain_generalization_sam.py \
  --dataset_id QIM+PMS+LSB+AHCM_0.5_1s --steg_algorithm SFFN \
  --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
  --data_root ${DASM_DATA_ROOT}/model_train/combined_multi \
  --batch_size 256 --embedding_rate 0.5 --eval_step 10 --epochs 100 --gpu 0 --save_model
# 多域学习 CCN（可以开始大数据集）
python model_domain_generalization.py \
  --dataset_id QIM+PMS+LSB+AHCM_0.5_1s --steg_algorithm CCN \
  --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
  --data_root ${DASM_DATA_ROOT}/model_train/combined_multi \
  --embedding_rate 0.5 --gpu 0 --save_model ;\
python model_domain_generalization.py \
  --dataset_id QIM+PMS+LSB+AHCM_0.5_1s --steg_algorithm SS-QCCN \
  --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
  --data_root ${DASM_DATA_ROOT}/model_train/combined_multi \
  --embedding_rate 0.5 --save_model
# 多域学习 SS-QCCN（可以开始大数据集）
python model_domain_generalization.py \
  --dataset_id QIM+PMS+LSB+AHCM_0.5_1s --steg_algorithm SS-QCCN \
  --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
  --data_root ${DASM_DATA_ROOT}/../data/model_train/er/combined_multi \
  --embedding_rate 0.5 --save_model



# 多域学习 SAM
python model_domain_generalization_sam.py \
    --dataset_id "QIM+PMS+LSB+AHCM_0.1_1s" --embedding_rate 0.1\
    --steg_algorithm Transformer --train_domains=QIM,PMS,LSB,AHCM --test_domains=QIM,PMS,LSB,AHCM \
    --use_sam --rho 0.05 --adaptive \
    --batch_size 280 --epochs 100 --save_model --eval_step 10 --gpu 0

# 使用C-SAM训练模型，必须要有参数contrast_lambda
# python model_domain_generalization_csam.py \
#     --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
#     --train_domains=QIM,PMS,LSB,AHCM --test_domains=QIM,PMS,LSB,AHCM \
#     --embedding_rate=0.5 --steg_algorithm=Transformer --batch_size 280 \
#     --rho 0.05 --contrast_lambda 1 --contrast_tau 0.07 \
#     --save_model --adaptive;

# 使用DBSM训练模型
# DBSM with adaptive only
python model_domain_generalization_dbsm.py \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains=QIM,PMS,AHCM --test_domains=LSB \
    --embedding_rate=0.5 --steg_algorithm=Transformer --batch_size 620 \
    --rho 0.05 --smooth_max_tau 0.1 \
    --save_model --adaptive;
python model_domain_generalization_dbsm.py \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains=QIM,PMS,AHCM --test_domains=LSB \
    --embedding_rate=0.5 --steg_algorithm=Transformer --batch_size 620 \
    --rho 0.05 --smooth_max_tau 0.1 --contrast_tau 0.07 --contrast_lambda 0.1 \
    --save_model --use_contrastive;
python model_domain_generalization_dbsm.py \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains=QIM,PMS,AHCM --test_domains=LSB \
    --embedding_rate=0.5 --steg_algorithm=Transformer --batch_size 360 \
    --rho 0.05 --smooth_max_tau 0.1 --contrast_tau 0.07 --contrast_lambda 0.1 \
    --save_model --adaptive --use_contrastive

# 指定嵌入率和模型
python visualize_training.py --mode compare --embedding_rate 0.3 --model_name LStegT




# 域泛化 使用FS-MDP_PMS
python model_domain_generalization.py --train_dataset=combined \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=QIM,LSB,AHCM \
    --test_domains=PMS --embedding_rate=0.5 --steg_algorithm=FS-MDP \
    --save_model --eval_step 5 --batch_size 1800 --lr=0.0001
# 域泛化 使用DVSF_QIM
python model_domain_generalization.py --train_dataset=combined \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=PMS,LSB,AHCM \
    --test_domains=QIM --embedding_rate=0.5 --steg_algorithm=DVSF \
    --save_model --eval_step 5 --batch_size 510 --lr=0.0001
# 域泛化 使用DVSF_PMS
python model_domain_generalization.py --train_dataset=combined \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=QIM,LSB,AHCM \
    --test_domains=PMS --embedding_rate=0.5 --steg_algorithm=DVSF \
    --save_model --eval_step 5 --batch_size 510 --lr=0.0001
# 域泛化 使用DVSF_LSB
python model_domain_generalization.py --train_dataset=combined \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=PMS,QIM,AHCM \
    --test_domains=LSB --embedding_rate=0.5 --steg_algorithm=DVSF \
    --save_model --eval_step 5 --batch_size 510 --lr=0.0001
# 域泛化 使用DVSF_AHCM
python model_domain_generalization.py --train_dataset=combined \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=PMS,QIM,LSB \
    --test_domains=AHCM --embedding_rate=0.5 --steg_algorithm=DVSF \
    --save_model --eval_step 5 --batch_size 510 --lr=0.0001
# 域泛化 使用DAEF-VS_QIM
python model_domain_generalization.py --train_dataset=combined \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=PMS,LSB,AHCM \
    --test_domains=QIM --embedding_rate=0.5 --steg_algorithm=DAEF-VS \
    --save_model --eval_step 5 --batch_size 300 --lr=0.0001
# 域泛化 使用DAEF-VS_PMS
python model_domain_generalization.py --train_dataset=combined \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=QIM,LSB,AHCM \
    --test_domains=PMS --embedding_rate=0.5 --steg_algorithm=DAEF-VS \
    --save_model --eval_step 5 --batch_size 300 --lr=0.0001
# 域泛化 使用DAEF-VS_LSB
python model_domain_generalization.py --train_dataset=combined \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=PMS,QIM,AHCM \
    --test_domains=LSB --embedding_rate=0.5 --steg_algorithm=DAEF-VS \
    --save_model --eval_step 5 --batch_size 300 --lr=0.0001
# 域泛化 使用DAEF-VS_AHCM
python model_domain_generalization.py --train_dataset=combined \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --train_domains=PMS,QIM,LSB \
    --test_domains=AHCM --embedding_rate=0.5 --steg_algorithm=DAEF-VS \
    --save_model --eval_step 5 --batch_size 300 --lr=0.0001








# 域泛化 仅测试
python ${DASM_ROOT}/model_domain_generalization.py --test_only \
    --steg_algorithm LStegT --train_dataset=combined \
    --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s --embedding_rate=0.5 \
    --train_domains=QIM,PMS,LSB --test_domains=AHCM \
    --result_path=${DASM_ROOT}/results_domain_gen/models_base/LStegT


# 使用SAM训练模型
python ${DASM_ROOT}/model_domain_gen_sam.py \
  --use_sam --rho 0.05 --adaptive \
  --steg_algorithm LStegT \
  --train_dataset combined \
  --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s.pkl \
  --embedding_rate 0.5 \
  --train_domains QIM,PMS,AHCM --test_domains LSB \
  --result_path ${DASM_ROOT}${DASM_ROOT}/results_domain_gen_sam


python ${DASM_ROOT}/model_domain_gen_sam.py \
  --use_sam --rho 0.05 --adaptive \
  --steg_algorithm Transformer \
  --train_dataset combined \
  --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s.pkl \
  --embedding_rate 0.5 \
  --train_domains QIM,PMS,LSB --test_domains AHCM \
  --result_path ${DASM_ROOT}${DASM_ROOT}/results_domain_gen_sam \
  --gpu 0 ; \
python ${DASM_ROOT}/model_domain_gen_sam.py \
  --use_sam --rho 0.05 --adaptive \
  --steg_algorithm Transformer \
  --train_dataset combined \
  --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s.pkl \
  --embedding_rate 0.5 \
  --train_domains PMS,LSB,AHCM --test_domains QIM \
  --result_path ${DASM_ROOT}${DASM_ROOT}/results_domain_gen_sam \
  --gpu 0


python ${DASM_ROOT}/model_domain_gen_sam.py \
  --use_sam --rho 0.05 --adaptive \
  --steg_algorithm Transformer \
  --train_dataset combined \
  --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s.pkl \
  --embedding_rate 0.5 \
  --train_domains QIM,LSB,AHCM --test_domains PMS \
  --result_path ${DASM_ROOT}${DASM_ROOT}/results_domain_gen_sam \
  --gpu 1 ; \
python ${DASM_ROOT}/model_domain_gen_sam.py \
  --use_sam --rho 0.05 --adaptive \
  --steg_algorithm Transformer \
  --train_dataset combined \
  --dataset_id=QIM+PMS+LSB+AHCM_0.5_1s.pkl \
  --embedding_rate 0.5 \
  --train_domains QIM,PMS,AHCM --test_domains LSB \
  --result_path ${DASM_ROOT}${DASM_ROOT}/results_domain_gen_sam \
  --gpu 1




#optimizer-based training
# ========== DISAM ==========
# DISAM - 任务1: AHCM+LSB+PMS → QIM
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name DISAM \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains AHCM,LSB,PMS \
    --test_domains QIM \
    --embedding_rate 0.5 \
    --batch_size 1000 \
    --save_model

# DISAM - 任务2: AHCM+LSB+QIM → PMS
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name DISAM \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains AHCM,LSB,QIM \
    --test_domains PMS \
    --embedding_rate 0.5 \
    --batch_size 1000 \
    --save_model

# DISAM - 任务3: AHCM+PMS+QIM → LSB
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name DISAM \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains AHCM,PMS,QIM \
    --test_domains LSB \
    --embedding_rate 0.5 \
    --batch_size 1000 \
    --save_model

# DISAM - 任务4: LSB+PMS+QIM → AHCM
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name DISAM \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains LSB,PMS,QIM \
    --test_domains AHCM \
    --embedding_rate 0.5 \
    --batch_size 1000 \
    --save_model

# ========== GAM ==========
# GAM - 任务1: AHCM+LSB+PMS → QIM
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name GAM \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains AHCM,LSB,PMS \
    --test_domains QIM \
    --embedding_rate 0.5 \
    --batch_size 8000 \
    --save_model

# GAM - 任务2: AHCM+LSB+QIM → PMS
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name GAM \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains AHCM,LSB,QIM \
    --test_domains PMS \
    --embedding_rate 0.5 \
    --batch_size 8000 \
    --save_model

# GAM - 任务3: AHCM+PMS+QIM → LSB
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name GAM \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains AHCM,PMS,QIM \
    --test_domains LSB \
    --embedding_rate 0.5 \
    --batch_size 8000 \
    --save_model

# GAM - 任务4: LSB+PMS+QIM → AHCM
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name GAM \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains LSB,PMS,QIM \
    --test_domains AHCM \
    --embedding_rate 0.5 \
    --batch_size 8000 \
    --save_model

# ========== GSAM ==========
# GSAM - 任务1: AHCM+LSB+PMS → QIM
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name GSAM \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains AHCM,LSB,PMS \
    --test_domains QIM \
    --embedding_rate 0.5 \
    --batch_size 10000 \
    --save_model

# GSAM - 任务2: AHCM+LSB+QIM → PMS
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name GSAM \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains AHCM,LSB,QIM \
    --test_domains PMS \
    --embedding_rate 0.5 \
    --batch_size 10000 \
    --save_model

# GSAM - 任务3: AHCM+PMS+QIM → LSB
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name GSAM \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains AHCM,PMS,QIM \
    --test_domains LSB \
    --embedding_rate 0.5 \
    --batch_size 10000 \
    --save_model

# GSAM - 任务4: LSB+PMS+QIM → AHCM
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name GSAM \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains LSB,PMS,QIM \
    --test_domains AHCM \
    --embedding_rate 0.5 \
    --batch_size 10000 \
    --save_model

# ========== Mixup ==========
# Mixup - 任务1: AHCM+LSB+PMS → QIM
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name Mixup \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains AHCM,LSB,PMS \
    --test_domains QIM \
    --embedding_rate 0.5 \
    --batch_size 16000 \
    --save_model

# Mixup - 任务2: AHCM+LSB+QIM → PMS
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name Mixup \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains AHCM,LSB,QIM \
    --test_domains PMS \
    --embedding_rate 0.5 \
    --batch_size 16000 \
    --save_model

# Mixup - 任务3: AHCM+PMS+QIM → LSB
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name Mixup \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains AHCM,PMS,QIM \
    --test_domains LSB \
    --embedding_rate 0.5 \
    --batch_size 16000 \
    --save_model

# Mixup - 任务4: LSB+PMS+QIM → AHCM
python ${DASM_ROOT}/model_domain_generalization_optimizers.py \
    --optimizer_name Mixup \
    --dataset_id QIM+PMS+LSB+AHCM_0.5_1s \
    --train_domains LSB,PMS,QIM \
    --test_domains AHCM \
    --embedding_rate 0.5 \
    --batch_size 16000 \
    --save_model











# 激活可视化
python ${DASM_ROOT}/neuron_visualization/model_domain_gen_vis.py \
    --visualize --train_dataset combined --dataset_id QIM+PMS+LSB+AHCM_0.5_1s.pkl --steg_algorithm LStegT --embedding_rate 0.5 --train_domains QIM,PMS,LSB --test_domains AHCM --result_path ${DASM_ROOT}${DASM_ROOT}/results_domain_gen_sam \
    --checkpoint_path "${DASM_ROOT}/results_domain_gen/models_base/LStegT/model_best_LStegT_QIM+PMS+LSB+AHCM_0.5_1s.pkl_1s_LSB_PMS_QIM_to_AHCM.pth.tar" \
    --viz_log_dir "./my_viz_logs" \
    --viz_export_format "both" --epochs 5 --batch_size 256

python ${DASM_ROOT}/neuron_visualization/model_domain_gen_vis.py \
    --visualize --viz_only --train_dataset combined --dataset_id QIM+PMS+LSB+AHCM_0.5_1s.pkl \
    --steg_algorithm LStegT --embedding_rate 0.5 --train_domains QIM,PMS,LSB --test_domains AHCM \
    --result_path ${DASM_ROOT}${DASM_ROOT}/results_domain_gen_sam \
    --checkpoint_path "${DASM_ROOT}/results_domain_gen/models_base/LStegT/model_best_LStegT_QIM+PMS+LSB+AHCM_0.5_1s.pkl_1s_LSB_PMS_QIM_to_AHCM.pth.tar" \
    --viz_log_dir "./my_viz_logs_matplotlib_fixed" --viz_export_format "both" --batch_size 64
