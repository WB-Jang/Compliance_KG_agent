# app/components/judge.py
from __future__ import annotations
import os, itertools, json
from typing import List, Dict, Type
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------
# 1) 온프레미스 LLM (llama-server OpenAI 호환)
# ---------------------------
LLM_BASE  = os.getenv("LLM_BASE",  "http://127.0.0.1:8081/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instruct")
LLM_KEY   = os.getenv("OPENAI_API_KEY", "not-needed")

def _llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        base_url=LLM_BASE,
        api_key=LLM_KEY,
    )

# ---------------------------
# 2) 구조화 출력 스키마 (Pydantic)
# ---------------------------
class JudgeVerdict(BaseModel):
    same: bool = Field(..., description="True if two entities are the same concept")
    why: str | None = Field(default=None, description="Short rationale")

# ---------------------------
# 3) 체인(LCEL): Prompt → LLM(structured)
# ---------------------------
JUDGE_SYS = (
    "You are a strict entity-equivalence judge for a regulatory knowledge graph. "
    "Decide if two entity records denote the SAME real-world concept within this legal context. "
    "Be conservative when definitions or applicability differ."
)

JUDGE_HUMAN = """Return a compact JSON with keys exactly: same (bool), why (str).
A:
{a}
B:
{b}"""

def _judge_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_SYS),
        ("human",  JUDGE_HUMAN),
    ])
    # 구조화 출력: Pydantic으로 강제
    return prompt | _llm().with_structured_output(JudgeVerdict)

# ---------------------------
# 4) 후보쌍 생성 (빠른 문자 유사도 블로킹)
# ---------------------------
def candidate_pairs(ents: List[Dict], ratio: int = 85):
    """
    문자 기반(rapidfuzz) 프리필터로 같은/유사 표기 엔티티들만 후보쌍으로 만든다.
    - ratio는 토큰정렬 유사도 기준(0~100)
    """
    pairs = []
    for a, b in itertools.combinations(ents, 2):
        scores = [fuzz.token_sort_ratio(a.get("canonical",""), b.get("canonical",""))]
        for x in a.get("aliases", []):
            for y in b.get("aliases", []):
                scores.append(fuzz.token_sort_ratio(x, y))
        if max(scores) >= ratio:
            pairs.append((a, b))
    return pairs

# ---------------------------
# 5) LLM-as-Judge (체인 사용)
# ---------------------------
def llm_equiv(a: Dict, b: Dict) -> bool:
    """
    LangChain 체인으로 llama-server에 질의 → JudgeVerdict(same, why) 구조로 수신.
    실패 시 보수적으로 False.
    """
    try:
        chain = _judge_chain()
        verdict: JudgeVerdict = chain.invoke({
            "a": json.dumps(a, ensure_ascii=False),
            "b": json.dumps(b, ensure_ascii=False),
        })
        return bool(verdict.same)
    except Exception:
        return False

# ---------------------------
# 6) 병합 루프
# ---------------------------
def merge_entities(ents: List[Dict]) -> List[Dict]:
    """
    - 후보쌍(candidate_pairs)으로 블로킹
    - 체인 기반 LLM-as-Judge로 동등 개념 판단
    - 동등하면 alias/source 병합 + 대표명(canonical) 단순 규칙으로 선택
    """
    ents = ents[:]  # shallow copy
    changed = True
    while changed:
        changed = False
        for a, b in candidate_pairs(ents):
            if llm_equiv(a, b):
                # 누락 키 보정
                a.setdefault("aliases", []); b.setdefault("aliases", [])
                a.setdefault("source",  []); b.setdefault("source",  [])

                # alias 병합(원래 canonical까지 alias로 흡수)
                a["aliases"] = sorted(set(
                    a["aliases"] + [a.get("canonical",""), b.get("canonical","")] + b["aliases"]
                ))

                # 대표명 선택(간단: 더 짧은 쪽; 필요시 규칙 고도화)
                if a.get("canonical") and b.get("canonical"):
                    a["canonical"] = min([a["canonical"], b["canonical"]], key=len)
                else:
                    a["canonical"] = a.get("canonical") or b.get("canonical")

                # source 병합
                a["source"] = sorted(set(a["source"] + b["source"]))

                # 타입/정의 충돌 등은 필요시 추가 규칙으로 조정
                # 예: a["type"] 우선, 없으면 b["type"] 채우기
                if not a.get("type") and b.get("type"):
                    a["type"] = b["type"]

                # b 제거 후 다시 루프 시작
                ents.remove(b)
                changed = True
                break
    return ents
