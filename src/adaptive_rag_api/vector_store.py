import logging
from typing import Any

import chromadb
import torch
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KananaEmbeddings(Embeddings):
    """Kanana 모델을 사용한 임베딩 클래스."""

    def __init__(
        self,
        model_name: str = "kakaocorp/kanana-nano-2.1b-embedding",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype="auto"
        )
        if self.device == "cuda":
            self.model = self.model.cuda()
        self.model.eval()
        logger.info(f"모델 '{model_name}' 초기화 완료")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """문서들을 임베딩합니다."""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """단일 쿼리를 임베딩합니다."""
        with torch.no_grad():
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512, padding=True
            )
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Kanana 모델에 필요한 pool_mask 추가
            attention_mask = inputs["attention_mask"]
            pool_mask = attention_mask.clone()
            inputs["pool_mask"] = pool_mask

            outputs = self.model(**inputs)

            # Kanana 임베딩 모델의 출력 구조에 맞게 처리
            if hasattr(outputs, "last_hidden_state"):
                embeddings = outputs.last_hidden_state.mean(dim=1)
            elif hasattr(outputs, "pooler_output"):
                embeddings = outputs.pooler_output
            elif hasattr(outputs, "embedding"):
                embeddings = outputs.embedding
            else:
                embeddings = (
                    outputs.embeddings if hasattr(outputs, "embeddings") else outputs
                )

            return embeddings.cpu().numpy().flatten().tolist()


class VectorStore:
    """ChromaDB 기반 벡터 저장소 클래스."""

    def __init__(
        self,
        persist_directory: str = "chroma",
        embedding_model_name: str = "kakaocorp/kanana-nano-2.1b-embedding",
        collection_name: str = "it_tech_news",
    ):
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.collection_name = collection_name
        self.embeddings = KananaEmbeddings(embedding_model_name)
        self._init_chroma_client()
        logger.info("벡터 저장소 컴포넌트 초기화 완료")

    def _init_chroma_client(self):
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB 클라이언트 초기화 완료: {self.persist_directory}")
        except Exception as e:
            logger.error(f"ChromaDB 클라이언트 초기화 실패: {e}")
            raise

    def create_collection(self, collection_name: str = None) -> chromadb.Collection:
        """컬렉션을 생성합니다."""
        try:
            if collection_name is None:
                collection_name = self.collection_name
            collection = self.client.get_or_create_collection(
                name=collection_name, metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"컬렉션 '{collection_name}' 생성/로드 완료")
            return collection
        except Exception as e:
            logger.error(f"컬렉션 생성 실패: {e}")
            raise

    def create_vectorstore(self, documents: list[Document]) -> bool:
        """문서들을 사용하여 벡터 저장소를 생성합니다."""
        try:
            logger.info(f"벡터 저장소 생성 시작: {len(documents)}개 문서")

            collection = self.create_collection(self.collection_name)

            # 기존 데이터 정리 (컬렉션을 완전히 삭제하고 재생성)
            try:
                self.client.delete_collection(name=self.collection_name)
                logger.info("기존 컬렉션 삭제")
                collection = self.create_collection(self.collection_name)
            except:
                # 컬렉션이 없는 경우는 무시
                pass

            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)
            ids = [f"doc_{i}" for i in range(len(documents))]

            collection.add(
                embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids
            )

            logger.info(f"벡터 저장소 생성 완료: {len(documents)}개 문서 저장")
            return True
        except Exception as e:
            logger.error(f"벡터 저장소 생성 실패: {e}")
            return False

    def delete_collection(self, collection_name: str = None) -> bool:
        """컬렉션을 삭제합니다."""
        try:
            if collection_name is None:
                collection_name = self.collection_name
            self.client.delete_collection(name=collection_name)
            logger.info(f"컬렉션 '{collection_name}' 삭제 완료")
            return True
        except Exception as e:
            logger.error(f"컬렉션 삭제 실패: {e}")
            return False

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        """유사도 기반 문서 검색을 수행합니다."""
        try:
            collection = self.create_collection(self.collection_name)
            query_embedding = self.embeddings.embed_query(query)

            results = collection.query(query_embeddings=[query_embedding], n_results=k)

            documents = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata) in enumerate(
                    zip(results["documents"][0], results["metadatas"][0])
                ):
                    documents.append(
                        Document(page_content=doc, metadata=metadata or {})
                    )

            return documents
        except Exception as e:
            logger.error(f"유사도 검색 실패: {e}")
            return []

    def search_similar(
        self, query: str, n_results: int = 5, collection_name: str = None
    ) -> list[dict[str, Any]]:
        """유사한 문서를 검색합니다."""
        try:
            if collection_name is None:
                collection_name = self.collection_name
            collection = self.create_collection(collection_name)
            query_embedding = self.embeddings.embed_query(query)
            results = collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"유사도 검색 실패: {e}")
            return []

    def get_collection_info(self, collection_name: str = None) -> dict[str, Any]:
        """컬렉션 정보를 반환합니다."""
        try:
            if collection_name is None:
                collection_name = self.collection_name
            collection = self.create_collection(collection_name)
            return {
                "collection_name": collection_name,
                "document_count": collection.count(),
                "persist_directory": self.persist_directory,
                "embedding_model": self.embedding_model_name,
            }
        except Exception as e:
            logger.error(f"컬렉션 정보 조회 실패: {e}")
            return {}

    def load_vectorstore(self) -> chromadb.Collection | None:
        """기존 벡터 저장소를 로드합니다."""
        try:
            collection = self.create_collection(self.collection_name)
            document_count = collection.count()
            if document_count > 0:
                logger.info(f"벡터 저장소 로드 완료: {document_count}개 문서")
                return collection
            else:
                logger.warning("벡터 저장소가 비어있습니다.")
                return None
        except Exception as e:
            logger.error(f"벡터 저장소 로드 실패: {e}")
            return None

    def add_documents(
        self, documents: list[Document], collection_name: str = None
    ) -> bool:
        """문서들을 벡터 저장소에 추가합니다."""
        try:
            if collection_name is None:
                collection_name = self.collection_name
            collection = self.create_collection(collection_name)
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)

            existing_count = collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

            collection.add(
                embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids
            )
            logger.info(f"문서 {len(documents)}개 추가 완료")
            return True
        except Exception as e:
            logger.error(f"문서 추가 실패: {e}")
            return False
