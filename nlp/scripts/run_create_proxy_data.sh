## w/ DPP

# nohup bash -x run_create_proxy_data.sh meta-llama/Llama-3.2-3B tau/commonsense_qa 5000 DPP 32 1e-5 > logs/log_ProxyData_llama_3.2_3b_commonsense_qa_5000_DPP.txt 2>&1 &
# nohup bash -x run_create_proxy_data.sh meta-llama/Llama-3.2-3B allenai/cosmos_qa 5000 DPP 32 1e-5 > logs/log_ProxyData_llama_3.2_3b_cosmos_qa_5000_DPP.txt 2>&1 &
# nohup bash -x run_create_proxy_data.sh meta-llama/Llama-3.2-3B allenai/social_i_qa 5000 DPP 32 1e-5 > logs/log_ProxyData_llama_3.2_3b_social_i_qa_5000_DPP.txt 2>&1 &

# nohup bash -x run_create_proxy_data.sh meta-llama/Llama-3.1-8B-Instruct tau/commonsense_qa 5000 DPP 32 1e-6 > logs/log_ProxyData_Llama3.1-8B-Instruct_commonsense_qa_5000_DPP.txt 2>&1 &
# nohup bash -x run_create_proxy_data.sh meta-llama/Llama-3.1-8B-Instruct allenai/cosmos_qa 5000 DPP 32 1e-6 > logs/log_ProxyData_Llama3.1-8B-Instruct_cosmos_qa_5000_DPP.txt 2>&1 &
# nohup bash -x run_create_proxy_data.sh meta-llama/Llama-3.1-8B-Instruct allenai/social_i_qa 5000 DPP 32 1e-6 > logs/log_ProxyData_Llama3.1-8B-Instruct_social_i_qa_5000_DPP.txt 2>&1 &

MODEL_PATH=${1}
DATASET=${2}
num_samples=${3:-5000}
DPP=${4:-"NODPP"}
BATCH_SIZE=${5:-16}
LR=${6:-1e-5}

DPP_STR=""
if [ "$DPP" == "DPP" ]; then 
    DPP_STR="--DPP"
fi

model_base_name=$(basename "${MODEL_PATH}")
ds_base_name=$(basename "${DATASET}")
job_id="ProxyData_${model_base_name}_${ds_base_name}_${num_samples}"

GPU_IDS="0,1,2,3,4,5,6,7"

cd ../..

CUDA_VISIBLE_DEVICES=${GPU_IDS} PYTHONPATH=. python nlp/nlp_create_proxy_data.py \
  --job_id ${job_id} \
  --model_path ${MODEL_PATH} \
  --local_data_path ${DATASET} \
  --num_samples ${num_samples} \
  --batch_size ${BATCH_SIZE} \
  --num_train_epochs 1 \
  --learning_rate ${LR} \
  ${DPP_STR}

echo "Job ID: ${job_id}"