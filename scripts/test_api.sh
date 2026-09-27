#!/bin/bash

# RAG API 테스트 스크립트

echo "=== RAG API 테스트 ==="

# 서비스가 실행 중인지 확인
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "RAG API가 실행 중이지 않습니다. 먼저 docker-compose up -d를 실행하세요."
    exit 1
fi

echo "1. 헬스체크 테스트"
curl -s http://localhost:8000/health | jq .

echo ""
echo "2. RAG 시스템 정보 테스트"
curl -s http://localhost:8000/rag/info | jq .

echo ""
echo "3. 채팅 완성 테스트"
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "query": "인공지능 기술의 최신 동향은 무엇인가요?",
    "max_iterations": 3
  }' | jq .

echo ""
echo "4. 벡터 검색 테스트"
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "머신러닝",
    "top_k": 3
  }' | jq .

echo ""
echo "=== 테스트 완료 ===" 