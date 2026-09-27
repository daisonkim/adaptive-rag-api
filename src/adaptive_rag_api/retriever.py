import logging
from dataclasses import dataclass
from typing import Any

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from adaptive_rag_api.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """검색 결과를 담는 데이터 클래스"""

    document: Document
    score: float
    rank: int
    rerank_score: float | None = None


class RerankedVectorRetriever(BaseRetriever):
    """벡터 검색 기반 Retriever 클래스"""

    vector_store: VectorStore
    reranker: CrossEncoderReranker | None = None
    top_k: int = 10
    rerank_top_k: int = 5
    similarity_threshold: float = 0.5

    def __init__(
        self,
        vector_store: VectorStore,
        reranker: CrossEncoderReranker | None = None,
        top_k: int = 10,
        rerank_top_k: int = 5,
        similarity_threshold: float = 0.5,
    ):
        """
        Args:
            vector_store: 벡터 저장소 인스턴스
            reranker: CrossEncoderReranker 인스턴스 (선택사항)
            top_k: 벡터 검색에서 가져올 문서 수
            rerank_top_k: 재순위화 후 반환할 문서 수
            similarity_threshold: 유사도 임계값
        """
        super().__init__(
            vector_store=vector_store,
            reranker=reranker,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
            similarity_threshold=similarity_threshold,
        )

        logger.info(
            f"RerankedVectorRetriever 초기화 완료 - top_k: {top_k}, rerank_top_k: {rerank_top_k}"
        )

    def _get_relevant_documents(self, query: str) -> list[Document]:
        try:
            documents = self.vector_store.similarity_search(query=query, k=self.top_k)

            if not documents:
                logger.warning(f"쿼리에 대한 검색 결과가 없습니다: {query}")
                return []

            if self.reranker:
                compressed_docs = self.reranker.compress_documents(
                    documents=documents, query=query
                )
                documents = compressed_docs[: self.rerank_top_k]
                logger.info(
                    f"CrossEncoderReranker 적용 완료 - 최종 문서 수: {len(documents)}"
                )
            else:
                documents = documents[: self.rerank_top_k]
                logger.info(f"벡터 검색만 사용 - 최종 문서 수: {len(documents)}")

            return documents

        except Exception as e:
            logger.error(f"문서 검색 실패: {e}")
            return []

    def get_relevant_documents_with_scores(
        self, query: str
    ) -> list[tuple[Document, float]]:
        """
        점수와 함께 관련 문서들을 검색합니다.

        Args:
            query: 사용자 질의

        Returns:
            (문서, 점수) 튜플 리스트
        """
        try:
            documents = self.vector_store.similarity_search(query=query, k=self.top_k)

            if not documents:
                return []

            if self.reranker:
                compressed_docs = self.reranker.compress_documents(
                    documents=documents, query=query
                )
                # CrossEncoderReranker는 점수를 직접 제공하지 않으므로 기본 점수 사용
                results = [
                    (doc, 1.0 - i / len(compressed_docs))
                    for i, doc in enumerate(compressed_docs[: self.rerank_top_k])
                ]
            else:
                results = [
                    (doc, 1.0 - i / len(documents))
                    for i, doc in enumerate(documents[: self.rerank_top_k])
                ]

            return results

        except Exception as e:
            logger.error(f"점수와 함께 문서 검색 실패: {e}")
            return []

    def search_with_metadata(
        self, query: str, filter_dict: dict[str, Any] | None = None
    ) -> list[SearchResult]:
        """
        메타데이터 필터링과 함께 검색을 수행합니다.

        Args:
            query: 사용자 질의
            filter_dict: 메타데이터 필터 딕셔너리

        Returns:
            SearchResult 리스트
        """
        try:
            documents = self.vector_store.similarity_search(
                query=query, k=self.top_k, filter_dict=filter_dict
            )

            if not documents:
                return []

            if self.reranker:
                compressed_docs = self.reranker.compress_documents(
                    documents=documents, query=query
                )
                # CrossEncoderReranker는 점수를 직접 제공하지 않으므로 기본 점수 사용
                search_results = [
                    SearchResult(doc, 1.0 - i / len(compressed_docs), i)
                    for i, doc in enumerate(compressed_docs[: self.rerank_top_k])
                ]
            else:
                search_results = [
                    SearchResult(doc, 1.0 - i / len(documents), i)
                    for i, doc in enumerate(documents[: self.rerank_top_k])
                ]

            return search_results

        except Exception as e:
            logger.error(f"메타데이터 필터링 검색 실패: {e}")
            return []


def create_retriever(
    vector_store: VectorStore,
    use_reranker: bool = True,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    top_k: int = 10,
    rerank_top_k: int = 5,
) -> RerankedVectorRetriever:
    """
    Retriever 인스턴스를 생성합니다.

    Args:
        vector_store: 벡터 저장소 인스턴스
        use_reranker: CrossEncoderReranker 사용 여부
        reranker_model: CrossEncoder 모델명
        top_k: 벡터 검색에서 가져올 문서 수
        rerank_top_k: 재순위화 후 반환할 문서 수

    Returns:
        RerankedVectorRetriever 인스턴스
    """
    reranker = None
    if use_reranker:
        try:
            cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
            reranker = CrossEncoderReranker(model=cross_encoder, top_n=rerank_top_k)
            logger.info(f"CrossEncoderReranker 생성 완료: {reranker_model}")
        except Exception as e:
            logger.warning(f"CrossEncoderReranker 생성 실패, 벡터 검색만 사용: {e}")

    retriever = RerankedVectorRetriever(
        vector_store=vector_store,
        reranker=reranker,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
    )

    logger.info(f"Retriever 생성 완료 - top_k: {top_k}, rerank_top_k: {rerank_top_k}")
    return retriever


if __name__ == "__main__":
    """테스트 코드"""
    from adaptive_rag_api.vector_store import VectorStore

    vector_store = VectorStore()
    vectorstore = vector_store.load_vectorstore()

    if vectorstore is None:
        logger.error("벡터 저장소를 로드할 수 없습니다.")
        exit(1)

    retriever = create_retriever(
        vector_store=vector_store, use_reranker=True, top_k=10, rerank_top_k=5
    )

    test_query = "인공지능 기술 동향"
    results = retriever._get_relevant_documents(test_query)

    print(f"검색 결과 수: {len(results)}")
    for i, doc in enumerate(results):
        print(f"\n--- 결과 {i + 1} ---")
        print(f"내용: {doc.page_content[:200]}...")
        if doc.metadata:
            print(f"메타데이터: {doc.metadata}")
