import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """프롬프트 템플릿 클래스."""

    name: str
    template: str
    description: str
    max_tokens: int = 4096


class PromptManager:
    """프롬프트 템플릿을 관리하는 클래스."""

    def __init__(self):
        self.templates = {}
        self._initialize_default_templates()
        logger.info(f"PromptManager 초기화 완료 - {len(self.templates)}개 템플릿")

    def _initialize_default_templates(self):
        self.add_template(
            PromptTemplate(
                name="rag_basic",
                template="""The following is relevant information retrieved from the IT technology news database.
Please answer the question based on this information.

=== Retrieved Information ===
{context}

=== Question ===
{query}

=== Answer ===
Please respond in the same language as the question.""",
                description="기본 RAG 프롬프트",
                max_tokens=3000,
            )
        )

        self.add_template(
            PromptTemplate(
                name="summary",
                template="""Please summarize the following IT technology news information.

=== Information ===
{context}

=== Summary ===
Please respond in the same language as the original text.""",
                description="정보 요약 프롬프트",
                max_tokens=2000,
            )
        )

        self.add_template(
            PromptTemplate(
                name="analysis",
                template="""Please analyze the following IT technology news information.

=== Information ===
{context}

=== Question ===
{query}

=== Analysis ===
Please respond in the same language as the question.""",
                description="정보 분석 프롬프트",
                max_tokens=3500,
            )
        )

        self.add_template(
            PromptTemplate(
                name="comparison",
                template="""Please compare and analyze the following IT technology news information.

=== Information ===
{context}

=== Comparative Analysis ===
Please respond in the same language as the original text.""",
                description="정보 비교 분석 프롬프트",
                max_tokens=3500,
            )
        )

        self.add_template(
            PromptTemplate(
                name="trend",
                template="""Please analyze the technology trends from the following IT technology news information.

=== Information ===
{context}

=== Technology Trend Analysis ===
Please respond in the same language as the original text.""",
                description="기술 트렌드 분석 프롬프트",
                max_tokens=3000,
            )
        )

        self.add_template(
            PromptTemplate(
                name="system_rag",
                template="""You are an IT technology news expert. 
Please provide accurate and useful answers based on the provided information.
Please respond in the same language as the user's question and include specific, practical information when possible.""",
                description="RAG용 시스템 프롬프트",
                max_tokens=100,
            )
        )

        self.add_template(
            PromptTemplate(
                name="system_general",
                template="You are an IT technology expert. Please provide accurate and useful answers to questions.",
                description="일반용 시스템 프롬프트",
                max_tokens=50,
            )
        )

        self.add_template(
            PromptTemplate(
                name="system_analyst",
                template="""You are an IT technology analyst. 
Please analyze the provided information and derive insights.
Please provide objective, data-driven responses.""",
                description="분석가용 시스템 프롬프트",
                max_tokens=100,
            )
        )

    def add_template(self, template: PromptTemplate):
        self.templates[template.name] = template

    def get_template(self, name: str) -> PromptTemplate | None:
        return self.templates.get(name)

    def format_prompt(self, template_name: str, **kwargs) -> str:
        template = self.get_template(template_name)
        if not template:
            logger.warning(f"템플릿 없음: {template_name}")
            return self._get_fallback_prompt(**kwargs)

        try:
            formatted = template.template.format(**kwargs)
            return formatted
        except KeyError as e:
            logger.error(f"프롬프트 오류: {e}")
            return self._get_fallback_prompt(**kwargs)

    def _get_fallback_prompt(self, **kwargs) -> str:
        context = kwargs.get("context", "")
        query = kwargs.get("query", "")

        return f"""다음 정보를 바탕으로 질문에 답변해주세요.

정보:
{context}

질문:
{query}

답변:"""

    def list_templates(self) -> dict[str, str]:
        return {name: template.description for name, template in self.templates.items()}

    def validate_template(self, template_name: str, **kwargs) -> bool:
        template = self.get_template(template_name)
        if not template:
            return False

        try:
            template.template.format(**kwargs)
            return True
        except KeyError:
            return False

    def get_template_info(self, template_name: str) -> dict[str, Any] | None:
        template = self.get_template(template_name)
        if not template:
            return None

        return {
            "name": template.name,
            "description": template.description,
            "max_tokens": template.max_tokens,
            "template": template.template,
        }


def create_prompt_manager() -> PromptManager:
    return PromptManager()


def create_rag_prompt(
    context: str, query: str, template_name: str = "rag_basic"
) -> str:
    manager = create_prompt_manager()
    return manager.format_prompt(template_name, context=context, query=query)


def create_summary_prompt(context: str) -> str:
    manager = create_prompt_manager()
    return manager.format_prompt("summary", context=context)


def create_analysis_prompt(context: str, query: str) -> str:
    manager = create_prompt_manager()
    return manager.format_prompt("analysis", context=context, query=query)


def get_system_prompt(prompt_type: str = "rag") -> str:
    manager = create_prompt_manager()

    if prompt_type == "rag":
        return manager.get_template("system_rag").template
    elif prompt_type == "general":
        return manager.get_template("system_general").template
    elif prompt_type == "analyst":
        return manager.get_template("system_analyst").template
    else:
        return manager.get_template("system_general").template


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    manager = create_prompt_manager()
    templates = manager.list_templates()
    print("사용 가능한 템플릿:")
    for name, desc in templates.items():
        print(f"  - {name}: {desc}")

    test_context = "인공지능 기술이 빠르게 발전하고 있습니다."
    test_query = "AI 기술의 현재 상황은?"

    rag_prompt = manager.format_prompt(
        "rag_basic", context=test_context, query=test_query
    )
    print(f"\nRAG 프롬프트:\n{rag_prompt}")

    summary_prompt = manager.format_prompt("summary", context=test_context)
    print(f"\n요약 프롬프트:\n{summary_prompt}")

    template_info = manager.get_template_info("rag_basic")
    print(f"\n템플릿 정보: {template_info}")
