export HF_ENDPOINT=https://hf-mirror.com

# conda activate yina-kv

# datasets=( "multi_news" "triviaqa" "2wikimqa" "lcc" )
# datasets=( "qasper" )
# datasets=( "multi_news" )
# datasets=( "multi_news" "qasper" "hotpotqa" "trec" "passage_count" "lcc" )
# datasets=( "narrativeqa" "multifieldqa_en" "2wikimqa" "musique" "gov_report" "qmsum" "triviaqa" "samsum" "passage_retrieval_en" "repobench-p" )
datasets=( "multi_news" "qasper" "hotpotqa" "trec" "passage_count" "lcc" "narrativeqa" "multifieldqa_en" "2wikimqa" "musique" "gov_report" "qmsum" "triviaqa" "samsum" "passage_retrieval_en" "repobench-p" )
# datasets=( "hotpotqa" )

config=( 256 512 1024 2048 )

for ds in "${datasets[@]}"; do
    echo dataset: $ds....
    # for l in "${config[@]}"; do
    #     echo dataset: $ds, config: $l
    #     CUDA_VISIBLE_DEVICES=0 python pred_snap.py --model mistral-7B-instruct-v0.2 --dataset $ds --compress_args_path ablation_c${l}_w32_k7_maxpool.json
    # done
    echo dataset: $ds, original
    CUDA_VISIBLE_DEVICES=0 python prompt_count_test.py --dataset $ds
    #CUDA_VISIBLE_DEVICES=0 python pred_snap.py --model mistral-7B-instruct-v0.1 --dataset $ds
done