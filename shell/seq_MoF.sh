export CUDA_VISIBLE_DEVICES=0,1,2,3
LOG_NAME=4card_seq_MoF
OUT_DIR=/mnt2/save_1M_seq_finetune/${LOG_NAME}

cd ../train/
python -m torch.distributed.run --nproc_per_node=4 --master_port=12600 train_MoF.py \
--config ../configs/exp/buffer.yaml \
--base_config ../configs/base_seqF.yaml \
--output_dir ${OUT_DIR} \
2>&1 | tee ../logger/${LOG_NAME}.log

