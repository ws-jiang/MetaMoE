##### Llama-3.2-3B

# bash -x run_lora_train_ours.sh meta-llama/Llama-3.2-3B tau/commonsense_qa 10000 ProxyData_Llama-3.2-3B_commonsense_qa_5000 500 DPP 16 10 > logs/log_MetaMoE_llama_3.2_3b_commonsense_qa_10000_ProxyDataNum500_DPP.txt
# bash -x run_lora_train_ours.sh meta-llama/Llama-3.2-3B allenai/cosmos_qa 10000 ProxyData_Llama-3.2-3B_cosmos_qa_5000 500 DPP 16 10 > logs/log_MetaMoE_llama_3.2_3b_cosmos_qa_10000_ProxyDataNum500_DPP.txt
# bash -x run_lora_train_ours.sh meta-llama/Llama-3.2-3B allenai/social_i_qa 10000 ProxyData_Llama-3.2-3B_social_i_qa_5000 500 DPP 16 10 > logs/log_MetaMoE_llama_3.2_3b_social_i_qa_10000_ProxyDataNum500_DPP.txt

## meta-llama/Llama-3.1-8B
# bash -x run_lora_train_ours.sh meta-llama/Llama-3.1-8B tau/commonsense_qa 10000 None 500 DPP 16 3 2e-4 > logs/log_MetaMoE_Llama3.1-8B_commonsense_qa_10000_ProxyDataNum500_DPP.txt 2>&1 &
# bash -x run_lora_train_ours.sh meta-llama/Llama-3.1-8B allenai/cosmos_qa 10000 None 500 DPP 16 3 2e-4 > logs/log_MetaMoE_Llama3.1-8B_cosmos_qa_10000_ProxyDataNum500_DPP.txt 2>&1 &
# bash -x run_lora_train_ours.sh meta-llama/Llama-3.1-8B allenai/social_i_qa 10000 None 500 DPP 16 3 2e-4 > logs/log_MetaMoE_Llama3.1-8B_social_i_qa_10000_ProxyDataNum500_DPP.txt 2>&1 &


MODEL_PATH=${1}
DATASET=${2}

num_samples=${3}
PROXY_DATA_PATH=${4}
PROXY_DATA_NUM_SAMPLES=${5}
DPP=${6:-"NODPP"}
batch_size=${7:-2}
num_epochs=${8:-10}
lr=${9:-1e-4}

DPP_STR=""
if [ "$DPP" == "DPP" ]; then
    DPP_STR="--DPP"
fi

model_base_name=$(basename "${MODEL_PATH}")
ds_base_name=$(basename "${DATASET}")
# Deterministic job_id (no timestamp) so the eval scripts can locate experts by config alone.
job_id="MetaMoE_${model_base_name}_${ds_base_name}_${num_samples}_ProxyDataNum${PROXY_DATA_NUM_SAMPLES}_${DPP}"

GPU_IDS="0,1,2,3,4,5,6,7"

cd ../..

CUDA_VISIBLE_DEVICES=${GPU_IDS} PYTHONPATH=. python nlp/nlp_train.py @nlp/configs/lora_csr_train.config \
  --job_id ${job_id} \
  --model_path ${MODEL_PATH} \
  --data_path ${DATASET} \
  --num_samples ${num_samples} \
  --proxy_data_path ${PROXY_DATA_PATH} \
  --proxy_data_num_samples ${PROXY_DATA_NUM_SAMPLES} \
  --num_train_epochs=${num_epochs} \
  --batch_size ${batch_size} \
  --learning_rate ${lr} \
  ${DPP_STR}

echo "Job ID: ${job_id}"