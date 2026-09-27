import logging
import time
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from adaptive_rag_api.adaptive_rag import create_adaptive_rag_langgraph
from adaptive_rag_api.rag_pipeline import create_rag_pipeline
from adaptive_rag_api.retriever import create_retriever
from adaptive_rag_api.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Adaptive RAG API",
    description="IT 기술 뉴스 기반 적응형 RAG API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RAGRequest(BaseModel):
    """RAG 요청 모델."""

    query: str = Field(
        ...,
        description="사용자 질의",
        example="인공지능 기술의 최신 동향은 무엇인가요?",
    )
    max_iterations: int | None = Field(
        default=3, description="최대 반복 횟수", ge=1, le=10, example=3
    )


class RAGResponse(BaseModel):
    """RAG 응답 모델."""

    success: bool = Field(description="처리 성공 여부")
    response: str = Field(description="생성된 응답")
    sources: list = Field(description="참조된 소스 문서들")
    iteration: int = Field(description="실제 반복 횟수")
    error: str | None = Field(default=None, description="오류 메시지")
    quality_check: dict[str, Any] | None = Field(
        default=None, description="품질 검사 결과"
    )


adaptive_rag = None


def initialize_rag_system():
    """RAG 시스템을 초기화합니다."""
    global adaptive_rag
    try:
        logger.info("RAG 시스템 초기화 시작")
        vector_store = VectorStore()

        collection_info = vector_store.get_collection_info()
        if not collection_info or collection_info.get("document_count", 0) == 0:
            logger.warning("벡터 저장소가 비어있거나 존재하지 않습니다.")
            logger.info("먼저 create_vector_db.py를 실행하여 벡터 DB를 생성해주세요.")
            raise Exception("벡터 저장소가 비어있거나 존재하지 않습니다")

        logger.info(
            f"벡터 저장소 로드 완료: {collection_info['document_count']}개 문서"
        )

        retriever = create_retriever(
            vector_store=vector_store, use_reranker=True, top_k=10, rerank_top_k=5
        )
        from adaptive_rag_api.generator import create_generator

        generator = create_generator()
        pipeline = create_rag_pipeline(retriever, generator)
        adaptive_rag = create_adaptive_rag_langgraph(pipeline, max_iterations=3)
        logger.info("RAG 시스템 초기화 완료")
    except Exception as e:
        logger.error(f"RAG 시스템 초기화 실패: {e}")
        raise


@app.on_event("startup")
async def startup_event():
    """앱 시작 시 RAG 시스템 초기화."""
    initialize_rag_system()


@app.get("/")
async def root():
    """루트 엔드포인트."""
    return {
        "message": "Adaptive RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "ready",
    }


@app.get("/health")
async def health_check():
    """헬스체크 엔드포인트."""
    return {
        "status": "healthy",
        "rag_system_ready": adaptive_rag is not None,
        "timestamp": time.time(),
    }


@app.post("/v1/chat/completions", response_model=RAGResponse)
async def chat_completions(request: RAGRequest):
    """OpenAI API 스타일의 채팅 완성 엔드포인트."""
    if adaptive_rag is None:
        raise HTTPException(
            status_code=503, detail="RAG 시스템이 초기화되지 않았습니다"
        )
    try:
        logger.info(f"채팅 완성 요청: {request.query[:50]}...")
        result = adaptive_rag.process_query(request.query)
        if not result["success"]:
            result["response"] = result.get(
                "response", "죄송합니다. 현재 질문에 대한 답변을 생성할 수 없습니다."
            )
            result["success"] = True
        response = RAGResponse(
            success=result["success"],
            response=result.get("response", ""),
            sources=result.get("sources", []),
            iteration=result.get("iteration", 0),
            error=result.get("error"),
            quality_check=result.get("quality_check"),
        )
        logger.info(f"채팅 완성 응답 생성 완료 - 성공: {response.success}")
        return response
    except Exception as e:
        logger.error(f"채팅 완성 처리 중 오류: {e}")
        return RAGResponse(
            success=True,
            response="죄송합니다. 현재 서비스에 일시적인 문제가 있습니다. 잠시 후 다시 시도해주세요.",
            sources=[],
            iteration=0,
            error=str(e),
            quality_check=None,
        )


@app.get("/rag/info")
async def rag_info():
    """RAG 시스템 정보 엔드포인트."""
    if adaptive_rag is None:
        raise HTTPException(
            status_code=503, detail="RAG 시스템이 초기화되지 않았습니다"
        )
    try:
        info = adaptive_rag.get_workflow_info()
        return {
            "system": "Adaptive RAG with LangGraph",
            "max_iterations": info["max_iterations"],
            "pipeline_info": info["pipeline_info"],
            "status": "ready",
        }
    except Exception as e:
        logger.error(f"RAG 정보 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")


@app.post("/search")
async def vector_search(request: dict):
    """벡터 검색 테스트 엔드포인트."""
    if adaptive_rag is None:
        raise HTTPException(
            status_code=503, detail="RAG 시스템이 초기화되지 않았습니다"
        )
    try:
        query = request.get("query", "")
        top_k = request.get("top_k", 3)
        logger.info(f"벡터 검색 요청: {query[:50]}...")
        retriever = adaptive_rag.pipeline.retriever
        results = retriever.get_relevant_documents(query)
        formatted_results = []
        for i, doc in enumerate(results[:top_k]):
            formatted_results.append(
                {
                    "rank": i + 1,
                    "content": doc.page_content[:200] + "..."
                    if len(doc.page_content) > 200
                    else doc.page_content,
                    "metadata": doc.metadata,
                }
            )
        return {
            "query": query,
            "results": formatted_results,
            "total_found": len(results),
        }
    except Exception as e:
        logger.error(f"벡터 검색 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"검색 오류: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "adaptive_rag_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
