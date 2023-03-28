#!/usr/bin/env bash

set -e
set -x

PY_ARGS=${@:1}

while true
do
    END_PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $END_PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done

python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:$END_PORT prompt_tune_squad_adapt.py --dist_train ${PY_ARGS}
