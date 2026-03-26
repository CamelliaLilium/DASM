DASM_ROOT="${DASM_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
DASM_DATA_ROOT="${DASM_DATA_ROOT:-${DASM_ROOT}/dataset}"

python model_dasm_tsne.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --contrast_tau 0.5 --rho 0.03 --seed 42 --epochs 100 --batch_size 2048 --gpu 0 --save_model \
    --tsne_enabled --tsne_output_dir ${DASM_ROOT}/tsne_results \
    --tsne_samples_per_class 1000 --tsne_perplexity 30 --tsne_n_iter 1000 --tsne_fontsize 18

python model_dasm_tsne.py \
    --dataset_id 'QIM+PMS+LSB+AHCM_0.5_1s' --embedding_rate 0.5 --steg_algorithm Transformer \
    --train_domains QIM,PMS,LSB,AHCM --test_domains QIM,PMS,LSB,AHCM \
    --use_dasm --contrast_tau 0.5 --rho 0.03 --seed 42 --epochs 100 --batch_size 4096 --gpu 0 --save_model \
    --tsne_enabled --tsne_output_dir ${DASM_ROOT}/tsne_results/samp1000_perp40_iter1000_v3 \
    --tsne_samples_per_class 1000 --tsne_perplexity 40 --tsne_n_iter 1000 --tsne_fontsize 18 \
    --tsne_interval 10 --tsne_save_data

# 方式1：指定数据文件
python ${DASM_ROOT}/tsne_results/replot_tsne.py --data_file tsne_data_epoch_5.npz

# 方式2：指定目录和epoch
python ${DASM_ROOT}/tsne_results/replot_tsne.py --epoch 100 \
    --data_dir ${DASM_ROOT}/tsne_results/samp1000_perp40_iter1000_v3 \
    --output_dir ${DASM_ROOT}/tsne_results/tsne_replot
