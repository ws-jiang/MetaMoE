### Create proxy data with DDP
# nohup bash -x run_create_proxy_data.sh vit_b32 flower102 10000 1 DPP > logs/log_ProxyData_MM_vit_b32_flower102_10000_DDP.txt 2>&1 &
# nohup bash -x run_create_proxy_data.sh vit_b32 pets 10000 1 DPP > logs/log_ProxyData_MM_vit_b32_pets_10000_DDP.txt 2>&1 &
# nohup bash -x run_create_proxy_data.sh vit_b32 eurosat 10000 1 DPP > logs/log_ProxyData_MM_vit_b32_eurosat_10000_DDP.txt 2>&1 &

# nohup bash -x run_create_proxy_data.sh vit_b16 flower102 10000 1 DPP 0 > logs/log_ProxyData_MM_vit_b16_flower102_10000_DDP.txt 2>&1 &
# nohup bash -x run_create_proxy_data.sh vit_b16 pets 10000 1 DPP 1 > logs/log_ProxyData_MM_vit_b16_pets_10000_DDP.txt 2>&1 &
# nohup bash -x run_create_proxy_data.sh vit_b16 eurosat 10000 1 DPP 2 > logs/log_ProxyData_MM_vit_b16_eurosat_10000_DDP.txt 2>&1 &


MODEL_PATH=${1}
DATASET=${2}
num_samples=${3:-10000}
num_epoch=${4:-1}
USE_DPP=${5:-"NODPP"}
GPU_IDS=${6:-"0"}

DPP_STR=""
if [ "${USE_DPP}" == "DPP" ]; then
  DPP_STR="--DPP"
fi


model_base_name=$(basename "${MODEL_PATH}")
ds_base_name=$(basename "${DATASET}")

job_id="ProxyData_${model_base_name}_${ds_base_name}_${num_samples}_${USE_DPP}"

# GPU_IDS="0,1,2,3,4,5,6,7"

cd ../..

CUDA_VISIBLE_DEVICES=${GPU_IDS} PYTHONPATH=. python cv/mm_create_proxy_data.py \
  --job_id ${job_id} \
  --arch ${MODEL_PATH} \
  --local_data_path ${DATASET} \
  --public_data_path imagenet \
  --num_samples ${num_samples} \
  --batch_size 128 \
  --num_train_epochs ${num_epoch} \
  ${DPP_STR} \

echo "Job ID: ${job_id}"