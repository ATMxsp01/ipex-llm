export HF_ENDPOINT=https://hf-mirror.com

# conda activate yina-kv

# datasets=( "multi_news" "triviaqa" "2wikimqa" "lcc" )
# datasets=( "qasper" )
# datasets=( "multi_news" )
# datasets=( "multi_news" "qasper" "hotpotqa" "trec" "passage_count" "lcc" )
datasets=( "qasper" "hotpotqa" "trec" "passage_count" "lcc" )
# datasets=( "narrativeqa" "multifieldqa_en" "2wikimqa" "musique" "gov_report" "qmsum" "triviaqa" "samsum" "passage_retrieval_en" "repobench-p" )
# datasets=( "passage_retrieval_en" )

# config=( 256 512 1024 2048 )
config=( 256 512 1024 )

for ds in "${datasets[@]}"; do
    echo dataset: $ds....
    for l in "${config[@]}"; do
        echo dataset: $ds, config: $l
        CUDA_VISIBLE_DEVICES=0 python pred_snap.py --model mistral-7B-instruct-v0.2 --dataset $ds --dtype fp16 --compress_args_path ablation_c${l}_w32_k7_maxpool.json
        #CUDA_VISIBLE_DEVICES=0 python pred_snap.py --model llama2-7b-chat-4k --dataset $ds --dtype fp16 --compress_args_path ablation_c${l}_w32_k7_maxpool.json
    done
    #echo dataset: $ds, original
    #CUDA_VISIBLE_DEVICES=0 python pred_snap.py --model mistral-7B-instruct-v0.2 --dataset $ds
    #CUDA_VISIBLE_DEVICES=0 python pred_snap.py --model llama2-7b-chat-4k --dataset $ds --dtype fp16
done