# nohup bash -x run_train_lora.sh vit_b32 flower102 10 0 0.01 500 DPP > logs/log_MM_MetaMoE_vit_b32_flower102_10_proxy500_DPP.txt 2>&1 &
# nohup bash -x run_train_lora.sh vit_b32 pets 10 1 0.01 500 DPP > logs/log_MM_MetaMoE_vit_b32_pets_10_proxy500_DPP.txt 2>&1 &
# nohup bash -x run_train_lora.sh vit_b32 eurosat 10 2 0.01 500 DPP > logs/log_MM_MetaMoE_vit_b32_eurosat_10_proxy500_DPP.txt 2>&1 &

# nohup bash -x run_train_lora.sh vit_b16 flower102 10 3 0.01 500 DPP > logs/log_MM_MetaMoE_vit_b16_flower102_10_proxy500_DPP.txt 2>&1 &
# nohup bash -x run_train_lora.sh vit_b16 pets 10 4 0.01 500 DPP > logs/log_MM_MetaMoE_vit_b16_pets_10_proxy500_DPP.txt 2>&1 &
# nohup bash -x run_train_lora.sh vit_b16 eurosat 10 5 0.01 500 DPP > logs/log_MM_MetaMoE_vit_b16_eurosat_10_proxy500_DPP.txt 2>&1 &

MODEL_PATH=${1:-vit_b32}
DATASET=${2:-"pets"}
NUM_EPOCHS=${3:-10}
GPU_IDS=${4:-"0"}
LR=${5:-0.01}
NUM_PROXY_SAMPLES=${6:-0}
DPP=${7:-"NODPP"}
BATCH_SIZE=${8:-128}
PROXY_ALL=${9:-"None"}

DPP_STR=""
if [ "${DPP}" == "DPP" ]; then
  DPP_STR="--DPP"
fi

# Deterministic job_id (no timestamp) so the eval scripts can locate experts by config alone.
job_id="MM_MetaMoE_${MODEL_PATH}_${DATASET}_${NUM_EPOCHS}_Proxy${NUM_PROXY_SAMPLES}_${DPP}"

cd ../..

CUDA_VISIBLE_DEVICES=${GPU_IDS} PYTHONPATH=. python cv/mm_train.py  \
  --job_id ${job_id} ${DPP_STR} \
  --arch ${MODEL_PATH} \
  --dataset ${DATASET} \
  --num_epochs=${NUM_EPOCHS} \
  --lr=${LR} \
  --batch_size ${BATCH_SIZE} \
  --num_proxy_samples=${NUM_PROXY_SAMPLES} \
  --proxy_all=${PROXY_ALL} \

echo "Job ID: ${job_id}"