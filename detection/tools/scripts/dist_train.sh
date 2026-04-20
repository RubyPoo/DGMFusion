
# #!/usr/bin/env bash

# set -x
# NGPUSLIST=$1
# NGPUS=$2
# PY_ARGS=${@:3}

# while true
# do
#     PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
#     status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
#     if [ "${status}" != "0" ]; then
#         break;
#     fi
# done
# echo $PORT


# CUDA_VISIBLE_DEVICES=${NGPUSLIST} python -m torch.distributed.launch --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}
# CUDA_VISIBLE_DEVICES=${NGPUSLIST} torchrun --nproc_per_node=${NGPUS} --rdzv_endpoint=localhost:${PORT} train.py --launcher pytorch ${PY_ARGS}

#!/usr/bin/env bash  

set -x  
NGPUSLIST=$1  
NGPUS=$2  
PY_ARGS=${@:3}  


# 查找一个可用的端口（这里我们假设不需要特别复杂的端口发现机制）  

PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))  

# 注意：在实际应用中，您可能想要一个更健壮的端口检查机制  

# 这里我们简单地假设随机选择的端口是可用的  

 

# 使用 torchrun 启动训练  

# 注意：我们不需要手动设置 CUDA_VISIBLE_DEVICES，因为 torchrun 会为每个进程设置  

# 但是，如果您需要为每个进程指定不同的 GPU，您可能需要编写一个包装脚本来处理这一点  

# 这里我们假设所有 GPU 都对单个进程可见，并且 NGPUS 控制了使用的 GPU 数量  

  

torchrun --nproc_per_node=${NGPUS} \  

    --master_port=${PORT} \  

    train.py --launcher none ${PY_ARGS}  

  

# 注意：  

# 1. 我们移除了 --rdzv_endpoint，因为 torchrun 通常不需要它。  

# 2. 我们将 --launcher 设置为 none，因为 torchrun 已经处理了分布式启动的逻辑。  

# 3. 如果 train.py 脚本内部需要处理 GPU 分配，请确保它能够在不设置 CUDA_VISIBLE_DEVICES 的情况下工作。  

#    如果 train.py 需要知道哪些 GPU 可用，它应该能够查询 torch.cuda.device_count() 和 torch.cuda.current_device()。  

# 4. 如果您的 train.py 脚本需要更复杂的 GPU 分配逻辑（例如，每个进程使用不同的 GPU 子集），  

#    您可能需要编写一个额外的脚本来为每个 torchrun 进程设置 CUDA_VISIBLE_DEVICES。