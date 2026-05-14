# Compute router embeddings from proxy-aligned experts (MetaMoE, eq. 7)

# w/ DPP
# nohup bash -x run_compute_embedding.sh vit_b32 pets 0 DPP > logs/log_Embedding_MM_MetaMoE_vit_b32_pets_DPP.txt 2>&1 &
# nohup bash -x run_compute_embedding.sh vit_b32 flower102 1 DPP > logs/log_Embedding_MM_MetaMoE_vit_b32_flower102_DPP.txt 2>&1 &
# nohup bash -x run_compute_embedding.sh vit_b32 eurosat 2 DPP > logs/log_Embedding_MM_MetaMoE_vit_b32_eurosat_DPP.txt 2>&1 &

# w/o DPP
# nohup bash -x run_compute_embedding.sh vit_b32 pets 0 NODPP > logs/log_Embedding_MM_MetaMoE_vit_b32_pets_NODPP.txt 2>&1 &
# nohup bash -x run_compute_embedding.sh vit_b32 flower102 1 NODPP > logs/log_Embedding_MM_MetaMoE_vit_b32_flower102_NODPP.txt 2>&1 &
# nohup bash -x run_compute_embedding.sh vit_b32 eurosat 2 NODPP > logs/log_Embedding_MM_MetaMoE_vit_b32_eurosat_NODPP.txt 2>&1 &

MODEL_PATH=${1:-vit_b32}
DATASET=${2:-"pets"}
GPU_IDS=${3:-"0"}
DPP=${4:-"NODPP"}

DPP_STR=""
if [ "${DPP}" == "DPP" ]; then
  DPP_STR="--DPP"
fi

cd ../..

CUDA_VISIBLE_DEVICES=${GPU_IDS} PYTHONPATH=. python cv/mm_compute_embedding.py \
  --arch ${MODEL_PATH} \
  --dataset ${DATASET} \
  ${DPP_STR}
