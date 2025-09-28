CUDA_VISIBLE_DEVICES=0 python3 -u main_dpo.py \
--dataset_name mimic_dpo \
--max_seq_length 100 \
--threshold 10 \
--batch_size 16 \
--epochs 5 \
--save_dir results/mimic_cxr_dpo_w_weight_ffilter_p1_newweight01 \
--step_size 1 \
--gamma 0.8 \
--seed 456789
# v3: beta 0.1
# v4: with new policy