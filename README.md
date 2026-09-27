# Adaptive RAG API - IT 기술 뉴스 기반 적응형 RAG 시스템

## 개요

IT 기술 뉴스 데이터를 기반으로 한 적응형 RAG(Retrieval-Augmented Generation) 시스템. 
Kanana Nano 2.1B 모델을 사용하여 고품질 AI 응답을 생성.

## 🏗️ 시스템 아키텍처

### 핵심 컴포넌트

#### 1. **RAG Pipeline** (`rag_pipeline.py`)
- **RAGPipeline**: 전체 RAG 파이프라인 조합
- **ContextBuilder**: 검색된 문서들을 컨텍스트로 구성
- **PromptManager**: 프롬프트 템플릿 관리

#### 2. **Vector Store** (`vector_store.py`)
- **VectorStore**: ChromaDB 기반 벡터 저장소
- **KananaEmbeddings**: Kanana 임베딩 모델 래퍼
- **similarity_search()**: 유사도 기반 문서 검색

#### 3. **Retriever** (`retriever.py`)
- **VectorRetriever**: 벡터 검색 기반 Retriever
- **CrossEncoderReranker**: 2단계 검색 (벡터 + 재순위화)
- **HybridRetriever**: 향후 확장용 하이브리드 검색

#### 4. **Generator** (`generator.py`)
- **VLLMGenerator**: vLLM 서버를 통한 LLM 호출
- **RAGGenerator**: RAG 응답 생성 및 품질 검사
- **할루시네이션 검사**: LLM 응답의 정확성 검증

#### 5. **Adaptive RAG** (`adaptive_rag.py`)
- **AdaptiveRAGLangGraph**: LangGraph 기반 적응형 처리
- **품질 검사**: 할루시네이션 및 관련성 검사
- **쿼리 재작성**: 품질 미달 시 자동 재시도

#### 6. **Data Processing** (`data_processor.py`)
- **DocumentLoader**: 마크다운 파일 로딩
- **DocumentSplitter**: 의미 기반 문서 분할
- **DataProcessor**: 전체 데이터 처리 파이프라인

## 🔧 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/health` | GET | 헬스체크 |
| `/rag/info` | GET | RAG 시스템 정보 |
| `/v1/chat/completions` | POST | AI 응답 생성 (RAG) |
| `/search` | POST | 벡터 검색 |
| `/docs` | GET | API 문서 (Swagger) |

## 📊 데이터 흐름

```
1. 사용자 질의 → 2. 벡터 검색 → 3. 컨텍스트 구성 → 4. 프롬프트 생성 → 5. LLM 추론 → 6. 품질 검사 → 7. 응답 반환
```

### 상세 과정
1. **벡터 검색**: `retriever.py`에서 유사도 기반 문서 검색
2. **컨텍스트 구성**: `context_builder.py`에서 검색된 문서들을 프롬프트에 포함
3. **프롬프트 생성**: `prompts.py`에서 템플릿 기반 프롬프트 구성
4. **LLM 추론**: `generator.py`에서 vLLM 서버를 통한 응답 생성
5. **품질 검사**: `adaptive_rag.py`에서 할루시네이션 및 관련성 검사

## 🚀 빠른 시작

모든 명령은 프로젝트 루트에서 실행한다.

### 1. 의존성 설치 및 벡터 DB 생성 (필수)
원본 데이터(`data/it_tech_news/*.md`)는 저장소에 포함하지 않는다. 마크다운 파일을 `data/it_tech_news/`에 넣은 뒤 실행한다.

```bash
uv sync
uv run python -m adaptive_rag_api.create_vector_db   # data/it_tech_news → ./chroma
```

### 2. 로컬 실행
```bash
./scripts/start_vllm_kanana.sh                       # vLLM 서버 (포트 8001)
uv run python -m adaptive_rag_api.main               # RAG API (포트 8000)
```

### 3. Docker Compose 실행
```bash
docker-compose up -d
./scripts/test_api.sh
```

- **vLLM Server** (`8001`): Kanana Nano 2.1B 모델 서빙
- **RAG API** (`8000`): FastAPI RAG 서비스
- 환경 변수 `VLLM_SERVER_URL`: vLLM 서버 URL (기본값: `http://localhost:8001`)
- 벡터 DB(`./chroma`)는 이미지에 포함하지 않고 `/app/chroma`로 마운트한다 (compose: bind mount, k8s: hostPath). 실행 전에 벡터 DB를 먼저 만든다.

### 4. Kubernetes 배포
```bash
kubectl apply -f deploy/k8s_manifest.yaml
```

## 🔍 사용 예시

### AI 응답 생성
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"query": "인공지능 기술의 최신 동향은?", "max_iterations": 3}'
```

### 벡터 검색
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "머신러닝", "top_k": 3}'
```

## 📁 프로젝트 구조

```
├── src/adaptive_rag_api/   # RAG API 패키지
│   ├── main.py             # FastAPI 애플리케이션
│   ├── create_vector_db.py # 벡터 DB 생성 스크립트
│   ├── rag_pipeline.py     # RAG 파이프라인 조합
│   ├── adaptive_rag.py     # LangGraph 기반 적응형 RAG
│   ├── retriever.py        # 벡터 검색 및 재순위화
│   ├── generator.py        # LLM 응답 생성
│   ├── vector_store.py     # ChromaDB 벡터 저장소
│   ├── context_builder.py  # 컨텍스트 구성
│   ├── prompts.py          # 프롬프트 템플릿 관리
│   └── data_processor.py   # 데이터 처리 파이프라인
├── data/it_tech_news/      # 원본 뉴스 마크다운 (git 미포함)
├── scripts/                # vLLM 실행·API 테스트 셸 스크립트
├── deploy/                 # Kubernetes 매니페스트
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```
