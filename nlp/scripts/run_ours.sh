## MetaMoE
# nohup bash -x run_ours.sh meta-llama/Llama-3.2-3B 10000 -1 1e-4 1 32 500 proxy 1.0 proxy DPP > logs/log_MetaMoE_llama_3.2_3b_Epoch1_BatchSize8_lambda0.7_DPP.txt 2>&1 &
# nohup bash -x run_ours.sh meta-llama/Llama-3.1-8B 10000 -1 1e-4 1 32 500 proxy 1.0 proxy DPP > logs/log_MetaMoE_llama_3.1_8b_Epoch1_BatchSize32_lambda1.0_DPP.txt 2>&1 &

BASE_INDEX_KEY=`date '+%Y%m%d_%H%M%S'`

MODEL_PATH=${1}
TRAIN_SAMPL_NUM=${2:-1000}
EVAL_SAMPL_NUM=${3:-1000}
LEARNING_RATE=${4:-1e-1}
NUM_TRAIN_EPOCHS=${5:-1}
BATCH_SIZE=${6:-4}
PROXY_DATA_NUM_SAMPLES=${7:-0}
WHICH_EXPERTS=${8:-"gt"}
LAMBDA_=${9:-0.5}
WHICH_DATA=${10:-"proxy"}
DPP=${11:-"NODPP"}
MAX_STEPS=${12:-"-1"}
SEED=${13:-0}

DPP_STR=""
if [ "${DPP}" == "DPP" ]; then
    DPP_STR="--DPP"
fi

model_base_name=$(basename "${MODEL_PATH}")
job_id="MetaMoE_${model_base_name}_${TRAIN_SAMPL_NUM}_${EVAL_SAMPL_NUM}_${LEARNING_RATE}_${NUM_TRAIN_EPOCHS}_${BATCH_SIZE}_${PROXY_DATA_NUM_SAMPLES}_${BASE_INDEX_KEY}_${DPP}"
lora_folder="outputs_train/${job_id}"

GPU_IDS="0,1,2,3,4,5,6,7"

cd ../..

CUDA_VISIBLE_DEVICES=${GPU_IDS} PYTHONPATH=. python nlp/nlp_eval_ours.py @nlp/configs/lora_csr_test.config \
  --job_id ${job_id} \
  --model_path ${MODEL_PATH} \
  --num_samples ${EVAL_SAMPL_NUM} \
  --learning_rate ${LEARNING_RATE} \
  --num_train_epochs ${NUM_TRAIN_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --proxy_data_num_samples ${PROXY_DATA_NUM_SAMPLES} \
  --lambda_ ${LAMBDA_} \
  --which_experts ${WHICH_EXPERTS} \
  --which_data ${WHICH_DATA} \
  --max_steps ${MAX_STEPS} \
  --seed ${SEED} \
  --warmup_steps 0 \
  ${DPP_STR}

cd nlp/scripts
