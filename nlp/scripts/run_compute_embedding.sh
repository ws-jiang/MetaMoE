######################################################
## compute embedding
# bash -x run_compute_embedding.sh meta-llama/Llama-3.2-3B tau/commonsense_qa 10000 None 500 NOBASE DPP PROXY
# bash -x run_compute_embedding.sh meta-llama/Llama-3.2-3B allenai/cosmos_qa 10000 None 500 NOBASE DPP PROXY
# bash -x run_compute_embedding.sh meta-llama/Llama-3.2-3B allenai/social_i_qa 10000 None 500 NOBASE DPP PROXY

# bash -x run_compute_embedding.sh meta-llama/Llama-3.1-8B tau/commonsense_qa 10000 None 500 NOBASE DPP PROXY
# bash -x run_compute_embedding.sh meta-llama/Llama-3.1-8B allenai/cosmos_qa 10000 None 500 NOBASE DPP PROXY
# bash -x run_compute_embedding.sh meta-llama/Llama-3.1-8B allenai/social_i_qa 10000 None 500 NOBASE DPP PROXY



MODEL_PATH=${1}
DATASET=${2}
EVAL_SAMPL_NUM=${3}
PROXY_DATA_PATH=${4:-None}
PROXY_DATA_NUM_SAMPLES=${5:-0}
USE_BASE=${6:-"NOBASE"}
DPP=${7:-"NODPP"}
PROXY=${8:-"NOPROXY"}

DPP_STR=""
if [ "${DPP}" == "DPP" ]; then
    DPP_STR="--DPP"
fi

PROXY_STR=""
if [ "${PROXY}" == "PROXY" ]; then
    PROXY_STR="--PROXY"
fi

BASE_STR=""
if [ "${USE_BASE}" == "BASE" ]; then
    BASE_STR="--use_base"
fi

GPU_IDS="0,1,2,3,4,5,6,7"

cd ../..

CUDA_VISIBLE_DEVICES=${GPU_IDS} PYTHONPATH=. python nlp/nlp_compute_router_embedding.py @nlp/configs/lora_csr_test.config \
  --data_path ${DATASET} \
  --model_path ${MODEL_PATH} \
  --num_samples ${EVAL_SAMPL_NUM} \
  --proxy_data_path ${PROXY_DATA_PATH} \
  --proxy_data_num_samples ${PROXY_DATA_NUM_SAMPLES} \
  ${BASE_STR} ${PROXY_STR} ${DPP_STR}
