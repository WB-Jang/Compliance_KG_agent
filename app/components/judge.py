# app/components/judge.py
from __future__ import annotations
import os, itertools, json
from typing import List, Dict, Type
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ---------------------------
# 1) 온프레미스 LLM (llama-server OpenAI 호환)
# ---------------------------

BASE_URL = "http://host.docker.internal:8080/v1" 

llm = ChatOpenAI(
    model="Midm-2.0-Base-Instruct-q4_0",           # 서버에서 인식 가능한 임의의 모델명
    base_url=BASE_URL,
    api_key="sk-local-anything", # 의미없는 토큰도 OK
    temperature=0.0,
    max_tokens=8192               # 서버 토큰 제한 고려
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
    return prompt | llm.with_structured_output(JudgeVerdict) # chain을 반환

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
            ents.pop(a)
            print(f'{a}를 pop하였습니다')
            ents.pop(b)
            print(f'{b}를 pop하였습니다')
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
        pairs = candidate_pairs(ents)
        for a, b in pairs:
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
                pairs_removed = pairs.remove(b)
                changed = True
                break
    ents.appned(pairs_removed)
    print(f'{pairs_removed}가 다시 all_e에 합산되었습니다') 
    return ents, pairs

def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def _names(e: Dict) -> set[str]:
    out = set()
    if e.get("canonical"): out.add(_norm(e["canonical"]))
    for a in e.get("aliases", []) or []:
        if a: out.add(_norm(a))
    return out

def build_name2canon(ents: List[Dict]) -> Dict[str, str]:
    m = {}
    for e in ents:
        can = _norm(e.get("canonical", ""))
        if not can: 
            continue
        for n in _names(e) | {can}:
            m[n] = can
    return m

def normalize_relations(relations: List[Dict], name2canon: Dict[str,str]) -> List[Dict]:
    out = []
    for r in relations:
        s = name2canon.get(_norm(r.get("s","")), _norm(r.get("s","")))
        o = name2canon.get(_norm(r.get("o","")), _norm(r.get("o","")))
        p = _norm(r.get("p",""))
        nr = dict(r)
        nr["s"], nr["o"], nr["p"] = s, o, p
        out.append(nr)
    return out

def _indices_involving(rel_norm: List[Dict], name_norm: str) -> List[int]:
    return [i for i, r in enumerate(rel_norm) if r["s"] == name_norm or r["o"] == name_norm]

def _direction_and_other(rn: Dict, center: str):
    if rn["s"] == center:  return "subj", rn["o"]
    if rn["o"] == center:  return "obj",  rn["s"]
    return "", ""

def _merge_evidence(dst: Dict, src: Dict) -> None: 
    e1 = set(dst.get("evidence", []) or [])
    e2 = set(src.get("evidence", []) or [])
    dst["evidence"] = sorted(e1 | e2)
    if "source" in dst or "source" in src:
        s1 = set(dst.get("source", []) or [])
        s2 = set(src.get("source", []) or [])
        dst["source"] = sorted(s1 | s2)


def dedupe_relations_simple(
    ents: List[Dict],
    relations: List[Dict],
    pairs: List[(Dict, Dict)],
    max_llm_checks: int = 200
) -> List[Dict]:
    name2canon = build_name2canon(ents)
    rel_norm = normalize_relations(relations, name2canon)

    to_delete = set()
    checks = 0

    for a, b in pairs:
        a_can = _norm(a.get("canonical","")); a_can = name2canon.get(a_can, a_can)
        b_can = _norm(b.get("canonical","")); b_can = name2canon.get(b_can, b_can)
        if not a_can or not b_can: 
            continue

        a_idxs = _indices_involving(rel_norm, a_can)
        b_idxs = _indices_involving(rel_norm, b_can)

        for i in a_idxs:
            if i in to_delete: continue
            r1n = rel_norm[i]

            for j in b_idxs:
                if j in to_delete or i == j: 
                    continue
                r2n = rel_norm[j]

                # 1) predicate 동일
                if r1n["p"] != r2n["p"]:
                    continue

                # 2) 방향 동일
                dir1, other1 = _direction_and_other(r1n, a_can)
                dir2, other2 = _direction_and_other(r2n, b_can)
                if not dir1 or not dir2 or dir1 != dir2:
                    continue

                # 3) 반대편 노드 동일
                if other1 != other2:
                    continue

                # ---- LLM 최종판정 ----
                if checks >= max_llm_checks:
                    break
                checks += 1

                if llm_equiv(relations[i], relations[j]):
                    # 더 많은 evidence 가진 쪽을 남김
                    ei = len(relations[i].get("evidence", []) or [])
                    ej = len(relations[j].get("evidence", []) or [])
                    keep, drop = (i, j) if ei >= ej else (j, i)
                    _merge_evidence(relations[keep], relations[drop])
                    to_delete.add(drop)

    return [r for k, r in enumerate(relations) if k not in to_delete]