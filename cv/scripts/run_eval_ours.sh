
## MetaMoE
# nohup bash -x run_eval_ours.sh 5 0.001 0 500 vit_b32 0.3 proxy DPP > logs/log_MM_MetaMoE_vit_b32_pets_proxy500_DPP_Lamada0.3_Epoch5.txt 2>&1 &
# nohup bash -x run_eval_ours.sh 5 0.001 1 500 vit_b16 0.3 proxy DPP > logs/log_MM_MetaMoE_vit_b16_pets_proxy500_DPP_Lamada0.3_Epoch5.txt 2>&1 &

NUM_EPOCHS=${1:-2}
LR=${2:-0.01}
GPU_IDS=${3:-"0"}
NUM_PROXY_SAMPLES=${4:-500}
MODEL_PATH=${5:-"vit_b32"}
LAMBDA=${6:-0.0}
WHICH_DATA=${7:-"proxy"}
DPP=${8:-"NODPP"}

DPP_STR=""
if [ "${DPP}" == "DPP" ]; then
  DPP_STR="--DPP"
fi


cd ../..

CUDA_VISIBLE_DEVICES=${GPU_IDS} PYTHONPATH=. python cv/mm_eval_mlp.py \
  --num_epochs ${NUM_EPOCHS} \
  --lr ${LR} \
  --num_proxy_samples ${NUM_PROXY_SAMPLES} \
  --arch ${MODEL_PATH} \
  --lambda_ ${LAMBDA} \
  --which_data ${WHICH_DATA} \
  ${DPP_STR}
