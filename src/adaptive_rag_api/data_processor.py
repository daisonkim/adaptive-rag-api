import logging
from pathlib import Path

from langchain.docstore.document import Document
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker

from adaptive_rag_api.vector_store import KananaEmbeddings

logger = logging.getLogger(__name__)


class DocumentLoader:
    """마크다운 문서 로더"""

    def __init__(self, data_path: str = "data/it_tech_news/"):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"데이터 경로를 찾을 수 없습니다: {self.data_path}")

    def load_markdown_files(self, max_files: int | None = None) -> list[Document]:
        """마크다운 파일들을 로드합니다."""
        md_files = list(self.data_path.glob("*.md"))

        if max_files:
            md_files = md_files[:max_files]

        logger.info(f"로드할 파일 수: {len(md_files)}")

        documents = []
        for file_path in md_files:
            try:
                loader = UnstructuredMarkdownLoader(str(file_path))
                docs = loader.load()

                for doc in docs:
                    doc.metadata["file_id"] = file_path.stem
                    doc.metadata["source"] = str(file_path)
                    documents.append(doc)

            except Exception as e:
                logger.warning(f"파일 로드 실패 {file_path}: {e}")

        logger.info(f"로드된 문서 수: {len(documents)}")
        return documents


class DataProcessor:
    """데이터 처리 클래스"""

    def __init__(
        self, embedding_model_name: str = "kakaocorp/kanana-nano-2.1b-embedding"
    ):
        logger.info("임베딩 모델 초기화 중...")
        self.embeddings = KananaEmbeddings(embedding_model_name)
        logger.info("SemanticChunker 초기화 중...")
        self.text_splitter = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
        logger.info("SemanticChunker 초기화 완료")
        logger.info("데이터 처리 컴포넌트 초기화 완료")

    def split_documents(self, documents: list[Document]) -> list[Document]:
        """의미 기반 텍스트 분할 수행"""
        logger.info("의미 기반 텍스트 분할 시작...")

        try:
            split_docs = self.text_splitter.split_documents(documents)

            chunk_sizes = [len(doc.page_content) for doc in split_docs]
            avg_chunk_size = sum(chunk_sizes) // len(chunk_sizes) if chunk_sizes else 0

            logger.info(f"청킹 완료: {len(documents)} → {len(split_docs)} 청크")
            logger.info(f"평균 청크 크기: {avg_chunk_size} 문자")

            return split_docs

        except Exception as e:
            logger.error(f"문서 분할 실패: {e}")
            raise

    def process_data(self, max_files: int | None = None) -> list[Document]:
        """전체 데이터 처리 파이프라인"""
        logger.info("=== 데이터 처리 시작 ===")
        logger.info("의미 기반 청킹을 사용하여 문서를 분할합니다.")

        if max_files:
            logger.info(f"테스트 모드: 처음 {max_files}개 파일만 로드")

        loader = DocumentLoader()

        documents = loader.load_markdown_files(max_files=max_files)
        logger.info(f"총 {len(documents)}개 문서 로드 완료")

        if not documents:
            logger.warning("로드된 문서가 없습니다.")
            return []

        logger.info(f"임베딩 모델: {self.embeddings.model_name}")

        split_documents = self.split_documents(documents)

        logger.info("=== 데이터 처리 완료 ===")
        return split_documents


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    processor = DataProcessor()
    documents = processor.process_data(max_files=5)
    logger.info(f"처리 완료: {len(documents)} 문서")

    if documents:
        logger.info("\n=== 청크 예시 ===")
        for i, doc in enumerate(documents[:3]):
            logger.info(f"\n청크 {i + 1}:")
            logger.info(f"문서 ID: {doc.metadata.get('file_id', 'N/A')}")
            logger.info(f"내용: {doc.page_content[:100]}...")
            logger.info(f"길이: {len(doc.page_content)} 문자")
