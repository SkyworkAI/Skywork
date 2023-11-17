for LOSS_DATA in zh_finance zh_general zh_government zh_movie zh_news zh_tech 
do 
    export HF_MODEL_PATH=YOUR_SKYWORK_HF_BASE_MODEL
    export FLAG=skywork-13b-base
    export DATA=$LOSS_DATA
    export BATCH_SIZE=16  
    mkdir -p prediction/$DATA/$FLAG
    python eval/eval_loss.py \
        -m $HF_MODEL_PATH --n-gpus 8 \
        -d data/eval_loss/$DATA.jsonl --data-type json -i text -b $BATCH_SIZE --max-tokens 4096 --max-samples 10000 \
        -o prediction/$DATA/$FLAG/result.txt
done 

