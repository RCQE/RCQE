# batch size 6 for 16 GB GPU

MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO

python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_test_cls.py  \
  --model_name_or_path /data/hzh22/CodeBERT/CodeReviewer/code/sh/els_model/token \
  --output_dir ./save \
  --load_model_path /data/hzh22/CodeBERT/CodeReviewer/code/sh/els_model/token \
  --output_dir empty \
  --eval_file /data/hzh22/CodeReviewProcesser/postprocess/data/ref-test.jsonl \
  --max_source_length 512 \
  --max_target_length 128 \
  --eval_batch_size 16 \
  --mask_rate 0.15 \
  --save_steps 4000 \
  --log_steps 100 \
  --train_steps 1200 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233
    # > test_reviewer2.log 2>&1 &