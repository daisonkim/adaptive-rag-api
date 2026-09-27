#!/bin/bash

# vLLM 서버 실행 스크립트 - Kanana Nano 2.1B Instruct
echo "vLLM 서버 시작 중..."

# 기본 설정
MODEL="kakaocorp/kanana-nano-2.1b-instruct"
HOST="0.0.0.0"
PORT="8001"
GPU_ID=1
# GPU 환경변수 설정
export CUDA_VISIBLE_DEVICES=$GPU_ID

# vLLM 서버 실행
vllm serve $MODEL \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code 