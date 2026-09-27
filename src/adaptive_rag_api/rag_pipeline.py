import logging
from typing import Any

from adaptive_rag_api.context_builder import ContextBuilder
from adaptive_rag_api.generator import RAGGenerator
from adaptive_rag_api.prompts import PromptManager, get_system_prompt
from adaptive_rag_api.retriever import RerankedVectorRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """전체 RAG 파이프라인을 관리하는 클래스."""

    def __init__(
        self,
        retriever: RerankedVectorRetriever,
        context_builder: ContextBuilder,
        prompt_manager: PromptManager,
        generator: RAGGenerator,
        default_template: str = "rag_basic",
        default_system_prompt: str = "rag",
    ):
        self.retriever = retriever
        self.context_builder = context_builder
        self.prompt_manager = prompt_manager
        self.generator = generator
        self.default_template = default_template
        self.default_system_prompt = default_system_prompt
        logger.info("RAG Pipeline 초기화 완료")

    def process_query(
        self,
        query: str,
        template_name: str | None = None,
        system_prompt_type: str | None = None,
        max_documents: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """쿼리를 처리하여 RAG 응답을 생성합니다."""
        try:
            logger.info(f"쿼리 처리 시작: {query[:50]}...")
            documents = self.retriever._get_relevant_documents(query)
            logger.info(f"검색된 문서 수: {len(documents)}")
            if not documents:
                return {
                    "success": False,
                    "error": "검색된 문서가 없습니다",
                    "response": "",
                    "sources": [],
                    "query": query,
                }
            if max_documents:
                self.context_builder.max_documents = max_documents
            if max_tokens:
                self.context_builder.max_tokens = max_tokens
            context_result = self.context_builder.build_context(documents, query)
            logger.info(
                f"컨텍스트 구성 완료: {context_result['document_count']}개 문서"
            )
            template_name = template_name or self.default_template
            prompt = self.prompt_manager.format_prompt(
                template_name, context=context_result["context"], query=query
            )
            system_prompt_type = system_prompt_type or self.default_system_prompt
            system_prompt = get_system_prompt(system_prompt_type)
            result = self.generator.generate_rag_response(
                query, documents, system_prompt
            )
            result.update(
                {
                    "query": query,
                    "template_used": template_name,
                    "system_prompt_type": system_prompt_type,
                    "context_info": {
                        "document_count": context_result["document_count"],
                        "token_count": context_result["token_count"],
                        "sources": context_result["sources"],
                    },
                }
            )
            logger.info("RAG 파이프라인 처리 완료")
            return result
        except Exception as e:
            logger.error(f"RAG 파이프라인 오류: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "",
                "sources": [],
                "query": query,
            }

    def process_summary(
        self, query: str, max_documents: int | None = None
    ) -> dict[str, Any]:
        """요약 처리를 수행합니다."""
        try:
            logger.info(f"요약 처리 시작: {query[:50]}...")
            documents = self.retriever._get_relevant_documents(query)
            if not documents:
                return {
                    "success": False,
                    "error": "검색된 문서가 없습니다",
                    "response": "",
                    "sources": [],
                }
            if max_documents:
                self.context_builder.max_documents = max_documents
            context_result = self.context_builder.build_context(documents, query)
            prompt = self.prompt_manager.format_prompt(
                "summary", context=context_result["context"], query=query
            )
            system_prompt = get_system_prompt("summary")
            result = self.generator.generate_rag_response(
                query, documents, system_prompt
            )
            result.update(
                {
                    "query": query,
                    "template_used": "summary",
                    "system_prompt_type": "summary",
                    "context_info": {
                        "document_count": context_result["document_count"],
                        "token_count": context_result["token_count"],
                        "sources": context_result["sources"],
                    },
                }
            )
            logger.info("요약 처리 완료")
            return result
        except Exception as e:
            logger.error(f"요약 처리 오류: {e}")
            return {"success": False, "error": str(e), "response": "", "sources": []}

    def process_analysis(
        self,
        query: str,
        template_name: str = "analysis",
        max_documents: int | None = None,
    ) -> dict[str, Any]:
        """분석 처리를 수행합니다."""
        try:
            logger.info(f"분석 처리 시작: {query[:50]}...")
            documents = self.retriever._get_relevant_documents(query)
            if not documents:
                return {
                    "success": False,
                    "error": "검색된 문서가 없습니다",
                    "response": "",
                    "sources": [],
                }
            if max_documents:
                self.context_builder.max_documents = max_documents
            context_result = self.context_builder.build_context(documents, query)
            prompt = self.prompt_manager.format_prompt(
                template_name, context=context_result["context"], query=query
            )
            system_prompt = get_system_prompt("analysis")
            result = self.generator.generate_rag_response(
                query, documents, system_prompt
            )
            result.update(
                {
                    "query": query,
                    "template_used": template_name,
                    "system_prompt_type": "analysis",
                    "context_info": {
                        "document_count": context_result["document_count"],
                        "token_count": context_result["token_count"],
                        "sources": context_result["sources"],
                    },
                }
            )
            logger.info("분석 처리 완료")
            return result
        except Exception as e:
            logger.error(f"분석 처리 오류: {e}")
            return {"success": False, "error": str(e), "response": "", "sources": []}

    def get_pipeline_info(self) -> dict[str, Any]:
        """파이프라인 정보를 반환합니다."""
        retriever_type = type(self.retriever).__name__

        reranker_info = {}
        if hasattr(self.retriever, "reranker") and self.retriever.reranker:
            reranker_info = {
                "type": type(self.retriever.reranker).__name__,
                "model": getattr(
                    self.retriever.reranker.model, "model_name", "unknown"
                ),
                "enabled": True,
            }
        else:
            reranker_info = {"type": "None", "model": "None", "enabled": False}

        return {
            "retriever": {
                "type": retriever_type,
                "top_k": getattr(self.retriever, "top_k", 10),
                "rerank_top_k": getattr(self.retriever, "rerank_top_k", 5),
                "reranker": reranker_info,
            },
            "context_builder": {
                "max_tokens": self.context_builder.max_tokens,
                "max_documents": self.context_builder.max_documents,
            },
            "prompt_manager": {
                "available_templates": list(self.prompt_manager.templates.keys()),
                "default_template": self.default_template,
            },
            "generator": {"type": type(self.generator).__name__},
            "defaults": {
                "template": self.default_template,
                "system_prompt": self.default_system_prompt,
            },
        }


def create_rag_pipeline(
    retriever: RerankedVectorRetriever,
    generator: RAGGenerator,
    max_tokens: int = 3000,
    max_documents: int = 5,
    default_template: str = "rag_basic",
    default_system_prompt: str = "rag",
) -> RAGPipeline:
    """RAG 파이프라인 인스턴스를 생성합니다."""
    context_builder = ContextBuilder(max_tokens=max_tokens, max_documents=max_documents)
    prompt_manager = PromptManager()
    return RAGPipeline(
        retriever,
        context_builder,
        prompt_manager,
        generator,
        default_template,
        default_system_prompt,
    )


if __name__ == "__main__":
    from adaptive_rag_api.generator import create_generator
    from adaptive_rag_api.retriever import create_retriever
    from adaptive_rag_api.vector_store import VectorStore

    logging.basicConfig(level=logging.INFO)

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

    pipeline = create_rag_pipeline(retriever, generator)

    info = pipeline.get_pipeline_info()
    print("RAG 파이프라인 정보:")
    print(f"  - Retriever: {info['retriever']['type']}")
    print(
        f"  - Context Builder: {info['context_builder']['max_tokens']} 토큰, {info['context_builder']['max_documents']} 문서"
    )
    print(f"  - Templates: {len(info['prompt_manager']['available_templates'])}개")
    print(f"  - Generator: {info['generator']['type']}")

    test_queries = [
        "인공지능 기술의 최신 동향은 무엇인가요?",
        "머신러닝과 딥러닝의 차이점은?",
        "클라우드 컴퓨팅의 장점은?",
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n=== 테스트 {i}: {query} ===")

        result = pipeline.process_query(query)

        if result["success"]:
            print(f"응답: {result['response'][:100]}...")
            print(f"소스: {len(result['context_info']['sources'])}개 문서")
        else:
            print(f"오류: {result['error']}")

        summary_result = pipeline.process_summary(query)
        if summary_result["success"]:
            print(f"요약: {summary_result['response'][:100]}...")
