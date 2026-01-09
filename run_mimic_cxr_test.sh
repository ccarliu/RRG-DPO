CUDA_VISIBLE_DEVICES=0 python3 -u main.py \
--dataset_name mimic_cxr \
--max_seq_length 100 \
--threshold 10 \
--batch_size 48 \
--epochs 30 \
--save_dir results/mimic_cxr \
--step_size 1 \
--gamma 0.8 \
--seed 456789 # >> test_ori6.txt
