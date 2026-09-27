import logging
from typing import Any

from langchain.schema import Document

logger = logging.getLogger(__name__)


class ContextBuilder:
    """검색된 문서들을 컨텍스트로 구성하는 클래스."""

    def __init__(self, max_tokens: int = 4000, max_documents: int = 5):
        self.max_tokens = max_tokens
        self.max_documents = max_documents
        logger.info(
            f"ContextBuilder 초기화 - max_tokens: {max_tokens}, max_documents: {max_documents}"
        )

    def build_context(self, documents: list[Document], query: str) -> dict[str, Any]:
        """검색된 문서들을 컨텍스트로 구성합니다."""
        if not documents:
            logger.warning("검색된 문서가 없습니다.")
            return {"context": "", "sources": [], "token_count": 0, "document_count": 0}

        limited_docs = documents[: self.max_documents]
        context_parts = []
        sources = []
        total_tokens = 0

        for i, doc in enumerate(limited_docs):
            content = doc.page_content.strip()
            metadata = doc.metadata or {}
            source = metadata.get("source", f"문서_{i + 1}")
            sources.append(source)
            doc_context = f"문서 {i + 1}:\n{content}\n"
            context_parts.append(doc_context)
            estimated_tokens = len(content) * 1.5
            total_tokens += estimated_tokens

            if total_tokens > self.max_tokens:
                logger.info(f"토큰 제한 도달: {total_tokens:.0f} > {self.max_tokens}")
                break

        context = "\n".join(context_parts)
        result = {
            "context": context,
            "sources": sources,
            "token_count": int(total_tokens),
            "document_count": len(context_parts),
        }

        logger.info(
            f"컨텍스트 구성 완료: {len(context_parts)}개 문서, {int(total_tokens)} 토큰"
        )
        return result

    def build_structured_context(
        self, documents: list[Document], query: str
    ) -> dict[str, Any]:
        """구조화된 컨텍스트를 구성합니다."""
        if not documents:
            return {
                "context": "",
                "sources": [],
                "token_count": 0,
                "document_count": 0,
                "structured_info": {},
            }

        limited_docs = documents[: self.max_documents]
        structured_info = {"topics": set(), "dates": set(), "companies": set()}
        context_parts = []
        sources = []
        total_tokens = 0

        for i, doc in enumerate(limited_docs):
            content = doc.page_content.strip()
            metadata = doc.metadata or {}
            source = metadata.get("source", f"문서_{i + 1}")
            sources.append(source)

            if "topic" in metadata:
                structured_info["topics"].add(metadata["topic"])
            if "date" in metadata:
                structured_info["dates"].add(metadata["date"])
            if "company" in metadata:
                structured_info["companies"].add(metadata["company"])

            doc_context = f"문서 {i + 1}:\n{content}\n"
            context_parts.append(doc_context)
            estimated_tokens = len(content) * 1.5
            total_tokens += estimated_tokens

            if total_tokens > self.max_tokens:
                break

        context = "\n".join(context_parts)
        structured_info = {k: list(v) for k, v in structured_info.items()}

        result = {
            "context": context,
            "sources": sources,
            "token_count": int(total_tokens),
            "document_count": len(context_parts),
            "structured_info": structured_info,
        }

        logger.info(f"구조화된 컨텍스트 구성 완료: {len(context_parts)}개 문서")
        return result

    def estimate_tokens(self, text: str) -> int:
        """텍스트의 토큰 수를 추정합니다."""
        return int(len(text) * 1.5)

    def truncate_context(self, context: str, max_tokens: int) -> str:
        """컨텍스트를 토큰 제한에 맞게 자릅니다."""
        estimated_tokens = self.estimate_tokens(context)

        if estimated_tokens <= max_tokens:
            return context

        max_chars = int(max_tokens / 1.5)
        truncated = context[:max_chars]
        last_period = truncated.rfind(".")
        if last_period > max_chars * 0.8:
            truncated = truncated[: last_period + 1]

        logger.info(
            f"컨텍스트 자름: {estimated_tokens} → {self.estimate_tokens(truncated)} 토큰"
        )
        return truncated


def create_context_builder(
    max_tokens: int = 4000, max_documents: int = 5
) -> ContextBuilder:
    """ContextBuilder 인스턴스를 생성합니다."""
    return ContextBuilder(max_tokens=max_tokens, max_documents=max_documents)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_documents = [
        Document(
            page_content="인공지능 기술이 빠르게 발전하고 있습니다.",
            metadata={"source": "ai_news.md", "topic": "AI", "date": "2024-01-01"},
        ),
        Document(
            page_content="머신러닝 모델의 성능이 향상되었습니다.",
            metadata={"source": "ml_news.md", "topic": "ML", "date": "2024-01-02"},
        ),
    ]

    builder = create_context_builder(max_tokens=1000, max_documents=3)
    result = builder.build_context(test_documents, "AI 기술에 대해 알려주세요")
    print(f"컨텍스트 길이: {len(result['context'])}")
    print(f"토큰 수: {result['token_count']}")
    print(f"소스: {result['sources']}")

    structured_result = builder.build_structured_context(
        test_documents, "AI 기술에 대해 알려주세요"
    )
    print(f"구조화된 정보: {structured_result['structured_info']}")
