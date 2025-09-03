# app/components/extractor.py
from __future__ import annotations
import os, json
from typing import List, Dict, Type
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# 1) 온프레미스 LLM 설정
# ---------------------------
# llama-server.exe (llama.cpp) 를 OpenAI 호환 모드로 띄운 경우:
#   예) --host 127.0.0.1 --port 8081 --api-server
# 기본값은 http://127.0.0.1:8081/v1 로 가정
LLM_BASE = os.getenv("LLM_BASE", "http://127.0.0.1:8081/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instruct")  # 서버에 로드한 모델 이름
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")      # llama-server는 보통 미검증. placeholder

def _llm() -> ChatOpenAI:
    """
    llama-server(OpenAI 호환)로 붙는 LangChain LLM.
    - base_url: http://host:port/v1
    - api_key: dummy 허용
    """
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        base_url=LLM_BASE,
        api_key=LLM_API_KEY,
    )

# ---------------------------
# 2) 구조화 스키마 (Pydantic)
# ---------------------------
class Distilled(BaseModel):
    doc_type: str | None = None
    title: str | None = None
    authority: str | None = None
    effective_date: str | None = None
    amendments: list[dict] = Field(default_factory=list)
    articles: list[dict] = Field(default_factory=list)

class Entities(BaseModel):
    entities: list[dict]

class Relations(BaseModel):
    relations: list[dict]

# ---------------------------
# 3) 프롬프트 템플릿
# ---------------------------
DISTILL_SYS = (
    "You are a precise information distiller for legal/regulatory texts. "
    "Fill the blueprint ONLY with facts grounded in the provided texts. "
    "Use null when absent. Keep arrays concise."
)
DISTILL_HUMAN = """Blueprint (JSON-like):
{blueprint}

Texts (grouped, semantically similar):
{joined_texts}"""

ENT_SYS = (
    "You extract DISTINCT legal entities from the distilled JSON. "
    "Each entity must represent a semantically UNIQUE concept."
)
ENT_HUMAN = """Return JSON with key 'entities' only:
[{{"id":"e1","canonical":"...","aliases":["..."],"type":"...","source":["..."]}}]

Distilled JSON:
{distilled}"""

REL_SYS = (
    "You extract relations (subject, predicate, object) between the given entities. "
    "Ground each relation in the distilled JSON; include minimal 'evidence' strings."
)
REL_HUMAN = """Return JSON with key 'relations' only:
[{{"s":"e1","p":"defines","o":"e2","evidence":["..."]}}]

Entities:
{entities}

Distilled JSON:
{distilled}"""

# ---------------------------
# 4) 체인 구성 (LCEL)
# ---------------------------
def _structured_chain(system: str, human: str, schema: Type[BaseModel]) -> Runnable:
    """
    - ChatPromptTemplate → ChatOpenAI(with_structured_output) 로 이어지는 체인
    - 입력: dict (프롬프트 변수)
    - 출력: Pydantic 모델 인스턴스
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", human)]
    )
    return prompt | _llm().with_structured_output(schema)

# 체인 인스턴스
_distill_chain = _structured_chain(DISTILL_SYS, DISTILL_HUMAN, Distilled)
_entities_chain = _structured_chain(ENT_SYS, ENT_HUMAN, Entities)
_relations_chain = _structured_chain(REL_SYS, REL_HUMAN, Relations)

# ---------------------------
# 5) 공개 함수 (파이프 접점)
# ---------------------------
def distill_group(blueprint: dict, texts: List[str], max_chars: int = 16000) -> dict:
    """
    코사인 블로킹으로 묶인 텍스트들(gtexts)을 증류 블루프린트에 맞게 구조화.
    - blueprint: iText2KG 스타일 JSON 블루프린트
    - texts: 그룹 내 원문 단위 텍스트들 (gtexts)
    """
    joined = "\n---\n".join(texts)[:max_chars]
    out: Distilled = _distill_chain.invoke(
        {"blueprint": json.dumps(blueprint, ensure_ascii=False),
         "joined_texts": joined}
    )
    return out.model_dump()

def extract_entities(distilled: dict) -> dict:
    """
    증류 결과(distilled JSON)를 기반으로 '고유 개념' 엔터티 목록만 추출.
    """
    out: Entities = _entities_chain.invoke(
        {"distilled": json.dumps(distilled, ensure_ascii=False)}
    )
    return out.model_dump()

def extract_relations(distilled: dict, entities: dict) -> dict:
    """
    확정된 엔터티 집합을 컨텍스트로 관계 추출.
    """
    out: Relations = _relations_chain.invoke(
        {
            "distilled": json.dumps(distilled, ensure_ascii=False),
            "entities": json.dumps(entities, ensure_ascii=False),
        }
    )
    return out.model_dump()
