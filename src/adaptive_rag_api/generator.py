import logging
import os
import time
from typing import Any

import requests
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMGenerator:
    """vLLM CLI 서버를 통한 LLM 생성 클래스."""

    def __init__(
        self,
        server_url: str | None = None,
        model_name: str = "kakaocorp/kanana-nano-2.1b-instruct",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout: int = 30,
    ):
        self.server_url = server_url or os.getenv(
            "VLLM_SERVER_URL", "http://localhost:8001"
        )
        self.server_url = self.server_url.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        logger.info(f"VLLM Generator 초기화 - 서버: {self.server_url}")
        logger.info(f"모델: {model_name}")

    def generate_response(
        self, prompt: str, system_prompt: str | None = None
    ) -> dict[str, Any]:
        """vLLM 서버에 요청을 보내 응답을 생성합니다."""
        try:
            request_data = {"model": self.model_name, "messages": []}
            if system_prompt:
                request_data["messages"].append(
                    {"role": "system", "content": system_prompt}
                )
            request_data["messages"].append({"role": "user", "content": prompt})
            request_data.update(
                {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "stream": False,
                }
            )
            start_time = time.time()
            response = requests.post(
                f"{self.server_url}/v1/chat/completions",
                json=request_data,
                timeout=self.timeout,
            )
            end_time = time.time()
            if response.status_code == 200:
                result = response.json()
                generated_text = result["choices"][0]["message"]["content"]
                logger.info(f"응답 생성 완료 - 소요시간: {end_time - start_time:.2f}초")
                return {
                    "success": True,
                    "response": generated_text,
                    "usage": result.get("usage", {}),
                    "model": result.get("model", self.model_name),
                    "processing_time": end_time - start_time,
                }
            else:
                logger.error(
                    f"vLLM 서버 오류: {response.status_code} - {response.text}"
                )
                return {
                    "success": False,
                    "error": f"서버 오류: {response.status_code}",
                    "response": "",
                }
        except requests.exceptions.Timeout:
            logger.error(f"vLLM 서버 타임아웃 (>{self.timeout}초)")
            return {"success": False, "error": "서버 타임아웃", "response": ""}
        except requests.exceptions.ConnectionError:
            logger.error(f"vLLM 서버 연결 실패: {self.server_url}")
            return {"success": False, "error": "서버 연결 실패", "response": ""}
        except Exception as e:
            logger.error(f"응답 생성 실패: {e}")
            return {"success": False, "error": str(e), "response": ""}

    def test_connection(self) -> bool:
        """vLLM 서버 연결을 테스트합니다."""
        try:
            response = requests.get(f"{self.server_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                available_models = [model["id"] for model in models.get("data", [])]
                logger.info(f"사용 가능한 모델: {available_models}")
                return self.model_name in available_models
            else:
                logger.error(f"모델 목록 조회 실패: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"서버 연결 테스트 실패: {e}")
            return False


class RAGGenerator:
    """RAG 시스템을 위한 생성 모듈."""

    def __init__(self, vllm_generator: VLLMGenerator):
        self.vllm_generator = vllm_generator
        logger.info("RAG Generator 초기화 완료")

    def generate_rag_response(
        self, query: str, documents: list[Document], system_prompt: str | None = None
    ) -> dict[str, Any]:
        """RAG 응답을 생성합니다."""
        try:
            if not documents:
                logger.warning("검색된 문서가 없습니다")
                return {
                    "success": False,
                    "error": "검색된 문서가 없습니다",
                    "response": "",
                    "sources": [],
                }
            context_parts = []
            for i, doc in enumerate(documents):
                content = doc.page_content.strip()
                metadata_info = ""
                if doc.metadata and doc.metadata.get("source"):
                    metadata_info = f"\n[출처: {doc.metadata.get('source')}]"
                doc_context = f"문서 {i + 1}:{metadata_info}\n{content}\n"
                context_parts.append(doc_context)
            context = "\n".join(context_parts)
            prompt = f"""다음은 IT 기술 뉴스 데이터베이스에서 검색된 관련 정보입니다.

다음 정보를 바탕으로 질문에 답변해주세요.

=== 검색된 정보 ===
{context}

=== 질문 ===
{query}

=== 답변 ===
"""
            if not system_prompt:
                system_prompt = """You are an IT technology news expert. 
Please provide accurate and useful answers based on the provided information.
Please respond in the same language as the user's question and include specific, practical information when possible."""
            result = self.vllm_generator.generate_response(prompt, system_prompt)
            sources = []
            for doc in documents:
                source_info = {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                }
                sources.append(source_info)
            result["sources"] = sources
            result["context_length"] = len(context)
            result["num_documents"] = len(documents)
            return result
        except Exception as e:
            logger.error(f"RAG 응답 생성 실패: {e}")
            return {"success": False, "error": str(e), "response": "", "sources": []}

    def generate_simple_response(
        self, query: str, system_prompt: str | None = None
    ) -> dict[str, Any]:
        """간단한 응답을 생성합니다."""
        return self.vllm_generator.generate_response(query, system_prompt)

    def check_hallucination(
        self, response: str, sources: list[Document]
    ) -> dict[str, Any]:
        """LLM을 사용하여 할루시네이션을 검사합니다."""
        if not sources:
            return {"is_hallucination": True, "confidence": 0.9, "reason": "소스 없음"}
        source_contents = []
        for i, doc in enumerate(sources[:3]):
            content = doc.page_content[:500]
            source_contents.append(f"Source {i + 1}: {content}")
        sources_text = "\n\n".join(source_contents)
        response_preview = response[:1000]
        prompt = f"""Please check if the following response is based on the provided sources.

Sources:
{sources_text}

Response:
{response_preview}

Please respond in the following JSON format:
{{
    "is_hallucination": true/false,
    "confidence": 0.0-1.0,
    "reason": "judgment basis"
}}

Consider the response as hallucination if it contains information not present in the sources or contradicts the source content."""
        system_prompt = """You are a text quality assessment expert. 
Please accurately determine whether the response is based on the provided sources.
Respond only in JSON format."""
        result = self.vllm_generator.generate_response(prompt, system_prompt)
        if not result["success"]:
            logger.warning(f"할루시네이션 검사 실패: {result['error']}")
            return {"is_hallucination": False, "confidence": 0.5, "reason": "검사 실패"}
        try:
            import json

            response_text = result["response"].strip()
            if "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                return {
                    "is_hallucination": parsed.get("is_hallucination", False),
                    "confidence": parsed.get("confidence", 0.5),
                    "reason": parsed.get("reason", "LLM 검사 결과"),
                }
            else:
                logger.warning("LLM 응답에서 JSON을 찾을 수 없음")
                return {
                    "is_hallucination": False,
                    "confidence": 0.5,
                    "reason": "JSON 파싱 실패",
                }
        except Exception as e:
            logger.error(f"할루시네이션 검사 결과 파싱 실패: {e}")
            return {
                "is_hallucination": False,
                "confidence": 0.5,
                "reason": f"파싱 오류: {str(e)}",
            }

    def check_relevance(self, query: str, response: str) -> dict[str, Any]:
        """LLM을 사용하여 관련성을 검사합니다."""
        prompt = f"""Please check if the following response is relevant to the question.

Question: {query}

Response: {response[:1000]}

Please respond in the following JSON format:
{{
    "is_relevant": true/false,
    "score": 0.0-1.0,
    "reason": "judgment basis"
}}

Determine whether the response directly answers the question or provides relevant information."""
        system_prompt = """You are a text relevance assessment expert.
Please accurately determine how relevant the response is to the question.
Respond only in JSON format."""
        result = self.vllm_generator.generate_response(prompt, system_prompt)
        if not result["success"]:
            logger.warning(f"관련성 검사 실패: {result['error']}")
            return {"is_relevant": False, "score": 0.0, "reason": "검사 실패"}
        try:
            import json

            response_text = result["response"].strip()
            if "{" in response_text and "}" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                return {
                    "is_relevant": parsed.get("is_relevant", False),
                    "score": parsed.get("score", 0.0),
                    "reason": parsed.get("reason", "LLM 검사 결과"),
                }
            else:
                logger.warning("LLM 응답에서 JSON을 찾을 수 없음")
                return {"is_relevant": False, "score": 0.0, "reason": "JSON 파싱 실패"}
        except Exception as e:
            logger.error(f"관련성 검사 결과 파싱 실패: {e}")
            return {
                "is_relevant": False,
                "score": 0.0,
                "reason": f"파싱 오류: {str(e)}",
            }


def create_generator(
    server_url: str | None = None,
    model_name: str = "kakaocorp/kanana-nano-2.1b-instruct",
    max_tokens: int = 1024,
    temperature: float = 0.7,
) -> RAGGenerator:
    """RAG Generator 인스턴스를 생성합니다."""
    vllm_server_url = server_url or os.getenv(
        "VLLM_SERVER_URL", "http://localhost:8001"
    )
    vllm_generator = VLLMGenerator(
        server_url=vllm_server_url,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not vllm_generator.test_connection():
        logger.warning(f"vLLM 서버 연결 실패: {vllm_server_url}")
        logger.warning("vLLM 서버가 실행 중인지 확인해주세요")
    rag_generator = RAGGenerator(vllm_generator)
    logger.info("RAG Generator 생성 완료")
    return rag_generator


if __name__ == "__main__":
    from adaptive_rag_api.retriever import create_retriever
    from adaptive_rag_api.vector_store import VectorStore

    vector_store = VectorStore()

    collection_info = vector_store.get_collection_info()
    if not collection_info or collection_info.get("document_count", 0) == 0:
        logger.error("벡터 저장소가 비어있거나 존재하지 않습니다.")
        logger.info("먼저 create_vector_db.py를 실행하여 벡터 DB를 생성해주세요.")
        exit(1)

    logger.info(f"벡터 저장소 로드 완료: {collection_info['document_count']}개 문서")

    retriever = create_retriever(
        vector_store=vector_store, use_reranker=True, top_k=10, rerank_top_k=5
    )
    generator = create_generator()
    test_query = "인공지능 기술의 최신 동향은 무엇인가요?"
    documents = retriever._get_relevant_documents(test_query)
    print(f"검색된 문서 수: {len(documents)}")
    result = generator.generate_rag_response(test_query, documents)
    if result["success"]:
        print(f"\n=== 질문 ===\n{test_query}")
        print(f"\n=== 답변 ===\n{result['response']}")
        print("\n=== 처리 정보 ===")
        print(f"소요시간: {result.get('processing_time', 0):.2f}초")
        print(f"컨텍스트 길이: {result.get('context_length', 0)} 문자")
        print(f"사용된 문서 수: {result.get('num_documents', 0)}개")
    else:
        print(f"응답 생성 실패: {result['error']}")
