import logging
from typing import Any, TypedDict

from langchain.schema import Document
from langgraph.graph import END, START, StateGraph

from adaptive_rag_api.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """그래프 상태를 정의합니다."""

    question: str
    generation: str
    documents: list[Document]
    iteration: int
    query_type: str
    quality_check: dict[str, Any]


class AdaptiveRAGLangGraph:
    """LangGraph를 사용한 적응형 RAG 시스템."""

    def __init__(self, pipeline: RAGPipeline, max_iterations: int = 3):
        self.pipeline = pipeline
        self.max_iterations = max_iterations
        self.workflow = self._build_workflow()
        logger.info("Adaptive RAG LangGraph 초기화 완료")

    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(GraphState)
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate", self._generate)
        workflow.add_node("quality_check", self._quality_check)
        workflow.add_node("rewrite_query", self._rewrite_query)
        workflow.add_conditional_edges(
            START,
            self._route_query,
            {"summary": "retrieve", "analysis": "retrieve", "general": "retrieve"},
        )
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "quality_check")
        workflow.add_conditional_edges(
            "quality_check",
            self._decide_next_step,
            {"success": END, "retry": "rewrite_query", "fail": END},
        )
        workflow.add_edge("rewrite_query", "retrieve")
        return workflow.compile()

    def _route_query(self, state: GraphState) -> str:
        question = state["question"]
        query_lower = question.lower()
        summary_keywords = ["요약", "정리", "핵심", "개요", "개념", "정의"]
        analysis_keywords = [
            "분석",
            "비교",
            "차이점",
            "장단점",
            "트렌드",
            "전망",
            "평가",
            "검토",
        ]
        for keyword in summary_keywords:
            if keyword in query_lower:
                logger.info(f"쿼리 라우팅: SUMMARY - '{keyword}' 키워드 감지")
                return "summary"
        for keyword in analysis_keywords:
            if keyword in query_lower:
                logger.info(f"쿼리 라우팅: ANALYSIS - '{keyword}' 키워드 감지")
                return "analysis"
        logger.info("쿼리 라우팅: GENERAL - 기본 처리")
        return "general"

    def _retrieve(self, state: GraphState) -> GraphState:
        question = state["question"]
        query_type = state.get("query_type", "general")
        logger.info(f"문서 검색 시작: {question[:50]}...")
        documents = self.pipeline.retriever._get_relevant_documents(question)
        logger.info(f"검색된 문서 수: {len(documents)}")
        return {**state, "documents": documents, "query_type": query_type}

    def _generate(self, state: GraphState) -> GraphState:
        question = state["question"]
        documents = state["documents"]
        query_type = state["query_type"]
        logger.info(f"응답 생성 시작: {query_type} 타입")
        if query_type == "summary":
            result = self.pipeline.process_summary(question)
        elif query_type == "analysis":
            result = self.pipeline.process_analysis(question)
        else:
            result = self.pipeline.process_query(question)
        if not result["success"]:
            logger.warning(f"응답 생성 실패: {result['error']}")
            return {
                **state,
                "generation": "",
                "quality_check": {"passed": False, "error": result["error"]},
            }
        return {
            **state,
            "generation": result["response"],
            "quality_check": {"passed": True, "response": result["response"]},
        }

    def _quality_check(self, state: GraphState) -> GraphState:
        question = state["question"]
        generation = state["generation"]
        documents = state["documents"]
        iteration = state.get("iteration", 0)
        hallucination_check = self._check_hallucination(generation, documents)
        relevance_check = self._check_relevance(question, generation)
        passed = (
            not hallucination_check["is_hallucination"]
            and relevance_check["is_relevant"]
        )
        quality_result = {
            "passed": passed,
            "hallucination_check": hallucination_check,
            "relevance_check": relevance_check,
            "iteration": iteration,
        }
        if not passed:
            if hallucination_check["is_hallucination"]:
                quality_result["failure_reason"] = (
                    f"할루시네이션 감지: {hallucination_check['reason']}"
                )
            else:
                quality_result["failure_reason"] = (
                    f"관련성 부족: {relevance_check['score']:.2f} < {relevance_check['threshold']}"
                )
        return {**state, "quality_check": quality_result}

    def _check_hallucination(
        self, response: str, sources: list[Document]
    ) -> dict[str, Any]:
        try:
            generator = self.pipeline.generator
            result = generator.check_hallucination(response, sources)
            logger.info(f"LLM 할루시네이션 검사 결과: {result}")
            return result
        except Exception as e:
            logger.error(f"LLM 할루시네이션 검사 실패: {e}")
            if not sources:
                return {
                    "is_hallucination": True,
                    "confidence": 0.9,
                    "reason": "소스 없음",
                }
            response_length = len(response)
            source_count = len(sources)
            if response_length > 500 and source_count < 2:
                return {
                    "is_hallucination": True,
                    "confidence": 0.7,
                    "reason": "긴 응답에 소스 부족",
                }
            if "모르겠습니다" in response or "확실하지 않습니다" in response:
                return {
                    "is_hallucination": False,
                    "confidence": 0.8,
                    "reason": "불확실성 표현",
                }
            return {
                "is_hallucination": False,
                "confidence": 0.6,
                "reason": "기본 검사 통과",
            }

    def _check_relevance(self, query: str, response: str) -> dict[str, Any]:
        try:
            generator = self.pipeline.generator
            result = generator.check_relevance(query, response)
            logger.info(f"LLM 관련성 검사 결과: {result}")
            return result
        except Exception as e:
            logger.error(f"LLM 관련성 검사 실패: {e}")
            query_words = set(query.lower().split())
            response_words = set(response.lower().split())
            overlap = len(query_words.intersection(response_words))
            relevance_score = overlap / max(len(query_words), 1)
            threshold = 0.6
            return {
                "is_relevant": relevance_score >= threshold,
                "score": relevance_score,
                "threshold": threshold,
            }

    def _decide_next_step(self, state: GraphState) -> str:
        quality_check = state["quality_check"]
        iteration = quality_check["iteration"]
        if quality_check["passed"]:
            logger.info("품질 검사 통과")
            return "success"
        if iteration >= self.max_iterations:
            logger.warning("최대 반복 횟수 도달")
            return "fail"
        logger.info(f"품질 검사 실패, 재시도: {iteration + 1}/{self.max_iterations}")
        return "retry"

    def _rewrite_query(self, state: GraphState) -> GraphState:
        original_query = state["question"]
        failure_reason = state["quality_check"]["failure_reason"]
        iteration = state["quality_check"]["iteration"]
        rewrite_patterns = {
            "specificity": "Please provide more specific details and concrete examples.",
            "simplification": "Please explain in simple terms with key points only.",
            "expansion": "Please include background information and broader context.",
            "clarification": "Please clarify and provide more detailed explanations.",
            "examples": "Please provide specific examples and practical applications.",
        }
        if (
            "hallucination" in failure_reason.lower()
            or "source" in failure_reason.lower()
        ):
            pattern = rewrite_patterns["specificity"]
        elif "relevance" in failure_reason.lower():
            pattern = rewrite_patterns["expansion"]
        elif "not useful" in failure_reason.lower():
            pattern = rewrite_patterns["clarification"]
        elif "not supported" in failure_reason.lower():
            pattern = rewrite_patterns["examples"]
        else:
            pattern = rewrite_patterns["simplification"]
        rewritten_query = f"{original_query} {pattern}"
        logger.info(f"쿼리 재작성: '{original_query}' → '{rewritten_query}'")
        return {**state, "question": rewritten_query, "iteration": iteration + 1}

    def process_query(self, query: str) -> dict[str, Any]:
        """적응형 쿼리 처리를 수행합니다."""
        logger.info(f"적응형 쿼리 처리 시작: {query[:50]}...")
        initial_state = {
            "question": query,
            "generation": "",
            "documents": [],
            "iteration": 0,
            "query_type": "",
            "quality_check": {},
        }
        try:
            final_state = None
            for chunk in self.workflow.stream(initial_state):
                for node_name, node_state in chunk.items():
                    logger.info(f"노드 실행: {node_name}")
                    final_state = node_state
            if final_state is None:
                return {
                    "success": False,
                    "error": "워크플로우 실행 실패",
                    "response": "",
                    "sources": [],
                }
            quality_check = final_state.get("quality_check", {})
            if quality_check.get("passed", False):
                return {
                    "success": True,
                    "response": final_state["generation"],
                    "sources": [
                        doc.metadata.get("source", "")
                        for doc in final_state["documents"]
                    ],
                    "iteration": quality_check.get("iteration", 0),
                    "quality_check": quality_check,
                }
            else:
                return {
                    "success": False,
                    "error": quality_check.get("failure_reason", "품질 검사 실패"),
                    "response": final_state["generation"],
                    "sources": [
                        doc.metadata.get("source", "")
                        for doc in final_state["documents"]
                    ],
                    "iteration": quality_check.get("iteration", 0),
                }
        except Exception as e:
            logger.error(f"워크플로우 실행 오류: {e}")
            return {"success": False, "error": str(e), "response": "", "sources": []}

    def get_workflow_info(self) -> dict[str, Any]:
        """워크플로우 정보를 반환합니다."""
        return {
            "max_iterations": self.max_iterations,
            "pipeline_info": self.pipeline.get_pipeline_info(),
        }


def create_adaptive_rag_langgraph(
    pipeline: RAGPipeline, max_iterations: int = 3
) -> AdaptiveRAGLangGraph:
    """LangGraph 기반 Adaptive RAG 인스턴스를 생성합니다."""
    return AdaptiveRAGLangGraph(pipeline, max_iterations)


if __name__ == "__main__":
    from adaptive_rag_api.generator import create_generator
    from adaptive_rag_api.rag_pipeline import create_rag_pipeline
    from adaptive_rag_api.retriever import create_retriever
    from adaptive_rag_api.vector_store import VectorStore

    logging.basicConfig(level=logging.INFO)
    vector_store = VectorStore()
    vectorstore = vector_store.load_vectorstore()
    if vectorstore is None:
        logger.error("벡터 저장소를 로드할 수 없습니다.")
        exit(1)
    retriever = create_retriever(
        vector_store=vector_store, use_reranker=True, top_k=10, rerank_top_k=5
    )
    generator = create_generator()
    pipeline = create_rag_pipeline(retriever, generator)
    adaptive_rag = create_adaptive_rag_langgraph(pipeline)
    info = adaptive_rag.get_workflow_info()
    print("LangGraph Adaptive RAG 정보:")
    print(f"  - 최대 반복: {info['max_iterations']}")
    print(f"  - 파이프라인: {info['pipeline_info']['retriever']['type']}")
    test_queries = [
        "인공지능 기술의 최신 동향은 무엇인가요?",
        "AI 기술들을 요약해주세요",
        "머신러닝과 딥러닝의 차이점을 분석해주세요",
    ]
    for i, query in enumerate(test_queries, 1):
        print(f"\n=== 테스트 {i}: {query} ===")
        result = adaptive_rag.process_query(query)
        if result["success"]:
            print(f"응답: {result['response'][:100]}...")
            print(f"반복 횟수: {result['iteration']}")
            print(f"품질 검사: {result['quality_check']['passed']}")
        else:
            print(f"오류: {result['error']}")
            print(f"반복 횟수: {result['iteration']}")
