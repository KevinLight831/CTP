export CUDA_VISIBLE_DEVICES=0
LOG_NAME=CTP
#LOG_NAME=CTP_ER
OUT_DIR=/mnt2/save_1M_seq_finetune/${LOG_NAME}

python eval.py \
--output_dir ${OUT_DIR} \
2>&1 | tee ./${LOG_NAME}.log
