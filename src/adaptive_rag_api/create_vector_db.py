import logging
import os

from adaptive_rag_api.data_processor import DataProcessor
from adaptive_rag_api.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorDBBuilder:
    """벡터 DB 생성을 담당하는 클래스"""

    def __init__(self):
        self.data_processor = None
        self.vector_store = None

    def initialize_components(self) -> None:
        """컴포넌트들을 초기화합니다."""
        try:
            logger.info("컴포넌트 초기화 중...")
            self.data_processor = DataProcessor()
            self.vector_store = VectorStore()
            logger.info("컴포넌트 초기화 완료")
        except Exception as e:
            logger.error(f"컴포넌트 초기화 실패: {e}")
            raise

    def process_data(self) -> list:
        """
        데이터를 처리합니다.

        Returns:
            처리된 문서 리스트
        """
        logger.info("1. 데이터 처리 중...")

        if not self.data_processor:
            raise ValueError("데이터 프로세서가 초기화되지 않았습니다.")

        documents = self.data_processor.process_data()

        if not documents:
            logger.warning("처리할 문서가 없습니다.")
            return []

        logger.info(f"처리된 문서 수: {len(documents)}")
        return documents

    def create_vectorstore(self, documents: list) -> bool:
        """벡터 저장소를 생성합니다."""
        logger.info("2. 벡터 저장소 생성 중...")
        if not self.vector_store:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다.")
        try:
            return self.vector_store.create_vectorstore(documents)
        except Exception as e:
            logger.error(f"벡터 저장소 생성 실패: {e}")
            return False

    def cleanup_if_needed(self) -> None:
        """필요시 기존 벡터 저장소를 정리합니다."""
        import shutil

        chroma_dir = "chroma"
        if os.path.exists(chroma_dir):
            logger.info("기존 ChromaDB 디렉토리 정리 중...")
            shutil.rmtree(chroma_dir)
            logger.info("ChromaDB 디렉토리 정리 완료")

    def verify_creation(self) -> None:
        """생성 결과를 확인합니다."""
        logger.info("3. 생성 결과 확인...")

        if not self.vector_store:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다.")

        info = self.vector_store.get_collection_info()
        logger.info(f"컬렉션 정보: {info}")

    def run_test_search(self) -> None:
        """테스트 검색을 수행합니다."""
        logger.info("4. 테스트 검색 수행...")

        if not self.vector_store:
            raise ValueError("벡터 저장소가 초기화되지 않았습니다.")

        test_queries = ["인공지능 기술", "클라우드 컴퓨팅", "블록체인", "머신러닝"]

        for query in test_queries:
            logger.info(f"\n검색 쿼리: '{query}'")
            try:
                results = self.vector_store.similarity_search(query, k=2)

                for i, doc in enumerate(results):
                    logger.info(f"  {i + 1}. {doc.page_content[:100]}...")
                    logger.info(f"     파일: {doc.metadata.get('file_id', 'N/A')}")

            except Exception as e:
                logger.error(f"검색 실패 ('{query}'): {e}")

    def print_summary(self) -> None:
        """생성 완료 요약을 출력합니다."""
        logger.info("\n=== 벡터 DB 생성 완료 ===")
        if self.vector_store:
            logger.info(f"저장 위치: {self.vector_store.persist_directory}")
            logger.info(f"컬렉션 이름: {self.vector_store.collection_name}")


def main():
    """벡터 DB 생성 메인 함수"""
    logger.info("=== IT Tech News 벡터 DB 생성 시작 ===")

    try:
        builder = VectorDBBuilder()

        builder.cleanup_if_needed()

        builder.initialize_components()

        documents = builder.process_data()
        if not documents:
            return

        if not builder.create_vectorstore(documents):
            return

        builder.verify_creation()

        builder.run_test_search()

        builder.print_summary()

        logger.info("벡터 DB 생성 성공적으로 완료!")

    except Exception as e:
        logger.error(f"벡터 DB 생성 중 오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
