MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
DATASET_FILENAME="./data/datasets/seed.jsonl"
NUM_GPUS=8

cd "$(realpath ..)"
for i in $(seq 0 $NUM_GPUS);
do
  CUDA_VISIBLE_DEVICES=$i python collect_gradients.py --model_name_or_path=$MODEL_NAME \
    --dataset_filename=$DATASET_FILENAME --device_split_size=$NUM_GPUS &
done
wait