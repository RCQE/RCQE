# batch size 6 for 16 GB GPU

mnt_dir="/home/codereview"

MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO

# bash test_nltk.sh

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_ref.py  \
  --model_name_or_path /data/hzh22/CodeBERT/CodeReviewer/code/sh/els_comment_model_onediff/checkpoints-last-6.0\
  --output_dir ./els_comment_model_onediff \
  --load_model_path /data/hzh22/CodeBERT/CodeReviewer/code/sh/els_comment_model_onediff/checkpoints-last-6.0 \
  --output_dir ./els_comment_model_onediff \
  --eval_file /data/hzh22/CodeReviewProcesser/postprocess/data2/test.jsonl \
  --max_source_length 512 \
  --max_target_length 200 \
  --eval_batch_size 12 \
  --mask_rate 0.15 \
  --save_steps 1800 \
  --beam_size 10 \
  --log_steps 100 \
  --train_steps 1200 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
