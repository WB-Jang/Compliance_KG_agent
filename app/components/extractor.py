from __future__ import annotations
import os, json
from typing import List, Dict, Type
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

BASE_URL = "http://host.docker.internal:8080/v1" 

llm = ChatOpenAI(
    model="Midm-2.0-Base-Instruct-q4_0",           # 서버에서 인식 가능한 임의의 모델명
    base_url=BASE_URL,
    api_key="sk-local-anything", # 의미없는 토큰도 OK
    temperature=0.5,
    max_tokens=8192               # 서버 토큰 제한 고려
)

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


DISTILL_SYS = ("""
당신은 법률/규제 문서를 정밀하게 요약·정제하는 역할을 합니다. 
    다음 지침을 반드시 지키세요: 
    - 제공된 텍스트에 근거하여 blueprint를 채우세요. 
    - 원문에 없는 경우 null로 표기하세요. 
    - 배열은 간결하게 유지하세요. 
    - 출력은 반드시 JSON 형식으로 하세요. 
    - 출력의 모든 값은 한국어로 작성하세요
""")

DISTILL_HUMAN = """
아래 blueprint를 기준으로 텍스트를 요약하여 채워 주세요.  
반드시 JSON으로만 출력하고, 값은 한국어로 작성해야 합니다.

Blueprint (JSON-like):
{blueprint}

텍스트 묶음 (의미적으로 유사한 그룹):
{joined_texts}

"""
ENT_SYS = ("""
당신은 한국어 법률 조문에서 핵심 엔티티를 추출하는 역할을 합니다.  
다음 지침을 반드시 지키세요:

- 출력은 JSON 형식으로만 하세요.
- canonical은 반드시 하나의 한국어 표현으로 지정하세요.
- aliases에는 약어, 다른 한국어 표현, 영어 번역명을 넣을 수 있습니다.
- 출력의 canonical, aliases, type, evidence 값은 반드시 한국어로 작성하세요.
- 조문 전체를 canonical로 두지 마세요. 원자적 개념(법률명, 조문번호, 정의된 용어, 기관명, 행위, 조건, 수치 등)만 추출하세요.
""")

ENT_HUMAN = """아래 텍스트에서 엔티티를 추출해 주세요.  
반드시 JSON으로만 출력하며, 키 이름은 정확히 'entities' 여야 합니다.  
출력 예시는 다음과 같습니다:

[{{
  "id":"e1",
  "canonical":"개인정보 보호법",
  "aliases":["개보법","Privacy Act"],
  "type":"법률",
  "source":["doc://개인정보보호법#제28조의2"],
  "evidence":["..."]
}},{{
  "id":"e2",
  "canonical":"제28조의2",
  "aliases":["가명정보의 처리"],
  "type":"조문",
  "source":["doc://개인정보보호법#제28조의2"],
  "evidence":["..."]
}}]

텍스트:
{distilled}
"""

REL_SYS = ("""
당신은 한국어 법률 조문에서 엔티티 간 관계를 추출하는 역할을 합니다.  
다음 지침을 반드시 지키세요:

- 출력은 JSON 형식으로만 하세요.
- 관계의 subject(s), predicate(p), object(o)는 모두 엔티티 canonical을 참조하세요.
- predicate 값은 한국어 동사 또는 명사구로 작성하세요. (예: "정의한다","허용한다","적용된다","발급한다","제한한다")
- evidence 값은 반드시 한국어 원문 일부를 그대로 인용하세요.
- 출력 JSON 이외의 설명은 하지 마세요.
""")
REL_HUMAN = """아래 엔티티와 텍스트를 참고하여 관계를 추출해 주세요.  
반드시 JSON으로만 출력하며, 키 이름은 정확히 'relations' 여야 합니다.  

엔티티:
{entities}

텍스트:
{distilled}
"""
# rels 추출에는 차라리 제로샷이 나은 듯. few shot은 그 예시에 너무 집착하는 듯한 모습을 보임
def _structured_chain(system: str, human: str, schema: Type[BaseModel]) -> Runnable:
    """
    - ChatPromptTemplate → ChatOpenAI(with_structured_output) 로 이어지는 체인
    - 입력: dict (프롬프트 변수)
    - 출력: Pydantic 모델 인스턴스
    """
    prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", human)]
    )
    return prompt | llm.with_structured_output(schema)

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
