# neo4j_io.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
from neo4j import GraphDatabase
from datetime import datetime
import re
import os

def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def uniq_list(xs: Optional[List[str]]) -> List[str]:
    if not xs:
        return []
    return sorted(set(x for x in xs if isinstance(x, str) and x.strip()))

def build_name_to_id(ents: List[Dict]) -> Dict[str, str]:
    """
    canonical/aliases 모든 표기를 -> 해당 entity id 로 매핑
    """
    m = {}
    for e in ents:
        eid = e.get("id")
        if not eid:
            continue
        can = e.get("canonical")
        if can:
            m[norm(can)] = eid
        for a in e.get("aliases", []) or []:
            if a:
                m[norm(a)] = eid
    return m

def sanitize_predicate(p: str) -> str:
    """
    관계 속성용 정규화(소문자, 공백 압축).
    (관계 타입으로 쓰지 않고 r.p_norm 속성으로만 보관)
    """
    return norm(p)

class Neo4jWriter:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # 고유 제약조건: :Entity(id) 유니크
    def ensure_constraints(self):
        cypher = """
        CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
        FOR (n:Entity) REQUIRE n.id IS UNIQUE
        """
        with self.driver.session() as sess:
            sess.run(cypher)

    # 엔터티 업서트 (간단 버전: 전달된 배열/맵을 그대로 세팅)
    def upsert_entities(self, ents: List[Dict], batch: int = 1000):
        rows = []
        now = datetime.utcnow().isoformat()

        for e in ents:
            row = {
                "id": e.get("id"),
                "canonical": e.get("canonical"),
                "type": e.get("type"),
                "aliases": uniq_list(e.get("aliases")),
                "source": uniq_list(e.get("source")),
                "evidence": uniq_list(e.get("evidence")),
                "attrs": e.get("attrs") or {},
                "updated_at": now,
            }
            if not row["id"]:
                continue
            rows.append({"props": row})

        if not rows:
            return

        q = """
        UNWIND $rows AS row
        MERGE (n:Entity {id: row.props.id})
        SET n += row.props
        """
        with self.driver.session() as sess:
            for i in range(0, len(rows), batch):
                sess.run(q, rows=rows[i:i+batch])

    # 관계 업서트
    def upsert_relations(
        self,
        ents: List[Dict],
        relations: List[Dict],
        batch: int = 1000,
        strict: bool = True,
    ):
        """
        relations의 s/o 가 '엔터티 id' 또는 'canonical/alias 문자열'일 수 있음.
        - id로 매칭 실패하면 문자열을 canonical/alias 매핑으로 id 해석 시도.
        - strict=True 이면 해석 실패한 관계는 건너뜀.
        """
        name2id = build_name_to_id(ents)
        id_set = {e["id"] for e in ents if e.get("id")}
        rows = []
        now = datetime.utcnow().isoformat()

        for r in relations:
            s_raw = str(r.get("s", "")).strip()
            o_raw = str(r.get("o", "")).strip()
            p_raw = str(r.get("p", "")).strip()

            # s/o가 이미 id이면 그대로, 아니면 이름->id 해석
            s_id = s_raw if s_raw in id_set else name2id.get(norm(s_raw))
            o_id = o_raw if o_raw in id_set else name2id.get(norm(o_raw))

            if not s_id or not o_id:
                if strict:
                    # 해석 실패 관계 스킵
                    continue
                else:
                    # 필요하면 느슨하게 노드도 함께 생성하는 로직을 넣을 수 있음
                    pass

            row = {
                "s_id": s_id,
                "o_id": o_id,
                "p": p_raw,
                "p_norm": sanitize_predicate(p_raw),
                "evidence": uniq_list(r.get("evidence")),
                "source": uniq_list(r.get("source")),
                "updated_at": now,
            }
            rows.append(row)

        if not rows:
            return

        # 동일 (s, p_norm, o) 조합을 하나로 MERGE
        q = """
        UNWIND $rows AS row
        MATCH (s:Entity {id: row.s_id})
        MATCH (o:Entity {id: row.o_id})
        MERGE (s)-[r:REL {p_norm: row.p_norm}]->(o)
        SET r.p = row.p,
            r.evidence = row.evidence,
            r.source = row.source,
            r.updated_at = row.updated_at
        """
        with self.driver.session() as sess:
            for i in range(0, len(rows), batch):
                sess.run(q, rows=rows[i:i+batch])
