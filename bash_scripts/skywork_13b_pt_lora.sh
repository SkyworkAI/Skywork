set -x
export WANDB_API_KEY=${WANDB_API_KEY:-YOUR_WANDB_API_KEY}
export WANDB_ENTITY=${WANDB_ENTITY:-YOUR_WANDB_ENTITY}
export WANDB_PROJECT=${WANDB_PROJECT:-YOUR_WANDB_PROJECT}

GPUS_PER_NODE=8
NODE_RANK=$([ -z "$RANK" ] && echo -n 0 || echo -n $RANK)
NNODES=$([ -z "$WORLD_SIZE" ] && echo -n 1 || echo -n $WORLD_SIZE)

MAX_STEP=1000  
LR=1e-4

DEBUG="false"
USE_LORA="true"
ZERO_STAGE=2

GLOBAL_BATCH_SIZE=32  # 8 * 4 
MICRO_BATCH_SIZE=1 
SAVE_STEP=1000 
EVAL_STEP=1000 
GRAD_ACC=$((${GLOBAL_BATCH_SIZE} / (${GPUS_PER_NODE} * $NNODES * ${MICRO_BATCH_SIZE}) ))

FLAG=Skywork-13B-Base-pt-zero${ZERO_STAGE}-peaklr${LR}-steps${MAX_STEP}-gbs${GLOBAL_BATCH_SIZE}

ROOT_PATH=${ROOT_PATH:-/data/user/your_name}
MODEL_PATH=${MODEL_PATH:-SKYWORK_13B_BASE_MODEL_PATH}

TRAIN_CACHE_DIR=${TRAIN_CACHE_DIR}
OUTPUT_DIR=$ROOT_PATH/run_output/skywork-13b-pt-trainer/$FLAG
LOAD_MODEL_PATH=$([ -z "$MODEL_PATH" ] && echo -n "$OUTPUT_DIR" || echo -n "$MODEL_PATH")

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_port 29501"
if [[ $NNODES -gt 1 ]]; then

    export NCCL_IB_HCA=mlx5
    export NCCL_IB_TC=136
    export NCCL_IB_SL=5
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_TIMEOUT=22
    export NCCL_SOCKET_IFNAME=bond0
    export NCCL_DEBUG=INFO
    NODE_RANK=$RANK
    if [ "$MASTER_ADDR" == "localhost" ] ; then $MASTER_ADDR=`hostname`; fi

    echo $MASTER_ADDR
    echo $MASTER_PORT
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
fi

if [ "$DEBUG" = "true" ]; then 
    EVAL_STEP=5
    GLOBAL_BATCH_SIZE=8
    GRAD_ACC=1

fi

DS_CONFIG=${DS_CONFIG:-train/ds_config/zero3_offload.json}

LOG_ARGS="
    --logging_steps 1 \
    --logging_dir tensorboard/$FLAG \
    --logging_strategy steps \
    --logging_first_step True \
    --report_to wandb \
    --run_name $FLAG
"

OUTPUT_ARGS="
    --save_strategy steps \
    --save_total_limit 500 \
    --save_steps $SAVE_STEP \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir 
"

TRAIN_ARGS="
    --do_train \
    --max_steps $MAX_STEP \
    --lr_scheduler_type constant_with_warmup \
    --learning_rate $LR \
    --weight_decay 0.1 \
    --warmup_steps 20 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --gradient_accumulation_steps $GRAD_ACC \
    --per_device_train_batch_size $MICRO_BATCH_SIZE 
"

EVAL_ARGS="
    --do_eval \
    --evaluation_strategy steps \
    --eval_steps $EVAL_STEP \
    --per_device_eval_batch_size 1 
"

LORA_ARGS=""
if [ "$USE_LORA" = "true" ]; then 
    LORA_ARGS="
        --use_lora \
        --lora_r 32 \
        --lora_alpha 8 \
        --lora_dropout 0.1 
    "
fi 


INPUT_ARGS="
    --model_name_or_path $LOAD_MODEL_PATH \
    --tokenizer_name_or_path $LOAD_MODEL_PATH \
    --train_cache_dir $TRAIN_CACHE_DIR 
"

EXTRA_ARGS="
    --seed 1234 \
    --deepspeed $DS_CONFIG \
    --gradient_checkpointing \
    --ddp_find_unused_parameters False \
    --preprocessing_num_workers 12 \
    --ddp_timeout 30000 \
    --torch_dtype bfloat16 \
    --bf16 \
    --load_in_kbits 16 
"

mkdir -p logs/$FLAG || True 
torchrun $DISTRIBUTED_ARGS train/run_pt.py \
    $LOG_ARGS \
    $OUTPUT_ARGS \
    $TRAIN_ARGS \
    $EVAL_ARGS \
    $INPUT_ARGS \
    $LORA_ARGS \
    $EXTRA_ARGS 2>&1 | tee -a logs/$FLAG/$RANK.log
 