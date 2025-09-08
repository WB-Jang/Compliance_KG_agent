from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from neo4j import GraphDatabase
from datetime import datetime
import json
import re
from zoneinfo import ZoneInfo

def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def uniq_list(xs: Optional[List[str]]) -> List[str]:
    if not xs:
        return []
    return sorted(set(x for x in xs if isinstance(x, str) and x.strip()))

ENTITY_ATTR_KEY_HINTS = ("code", "number")  # 있으면 동일성 키에 포함

def entity_key(e: Dict) -> str:
    c = norm(e.get("canonical", ""))
    t = e.get("type", "") or ""
    parts = [c, "||", t]
    attrs = e.get("attrs") or {}
    for k in ENTITY_ATTR_KEY_HINTS:
        if k in attrs and attrs[k]:
            parts += ["||", f"{k}=", str(attrs[k])]
    return "".join(parts)

def rel_key(r: Dict, id2canon_norm: Dict[str, str] | None = None) -> str:
    # r["s"], r["o"]가 id거나 이름일 수 있음 -> 가능하면 canonical로 치환
    s_raw = str(r.get("s", ""))
    o_raw = str(r.get("o", ""))
    p_raw = str(r.get("p", r.get("p_norm","")))
    p_norm = norm(r.get("p_norm", p_raw))

    def to_canon_norm(x: str) -> str:
        if id2canon_norm and x in id2canon_norm:
            return id2canon_norm[x]
        return norm(x)

    s = to_canon_norm(s_raw)
    o = to_canon_norm(o_raw)
    return f"{s}||{p_norm}||{o}"


class Neo4jIO:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def ensure_constraints(self):
        q = """CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
               FOR (n:Entity) REQUIRE n.id IS UNIQUE"""
        with self.driver.session() as s:
            s.run(q)

    
    def read_entities(self) -> List[Dict]:
        q = "MATCH (n:Entity) RETURN n"
        out = []
        with self.driver.session() as s:
            for rec in s.run(q):
                n = rec["n"]
                out.append({
                    "id": n.get("id"),
                    "canonical": n.get("canonical"),
                    "aliases": n.get("aliases") or [],
                    "type": n.get("type"),
                    "source": n.get("source") or [],
                    "evidence": n.get("evidence") or [],
                    "attrs": n.get("attrs") or {},
                })
        return out

    def read_relations(self) -> List[Dict]:
        q = """MATCH (s:Entity)-[r:REL]->(o:Entity)
               RETURN s.id AS s_id, s.canonical AS s_canon,
                      r.p AS p, r.p_norm AS p_norm, r.evidence AS evidence, r.source AS source,
                      o.id AS o_id, o.canonical AS o_canon"""
        out = []
        with self.driver.session() as s:
            for rec in s.run(q):
                out.append({
                    "s_id": rec["s_id"],
                    "s": rec["s_canon"],      # 비교 편의를 위해 canonical로 채움
                    "p": rec["p"],
                    "p_norm": rec["p_norm"],
                    "evidence": rec["evidence"] or [],
                    "source": rec["source"] or [],
                    "o_id": rec["o_id"],
                    "o": rec["o_canon"],
                })
        return out

    def upsert_entities(self, ents: List[Dict], batch: int = 500):
        if not ents:
            return
        rows = []
        now = datetime.utcnow().isoformat()
        for e in ents:
            rows.append({"props": {
                "id": e.get("id"),
                "canonical": e.get("canonical"),
                "type": e.get("type"),
                "aliases": uniq_list(e.get("aliases")),
                "source": uniq_list(e.get("source")),
                "evidence": uniq_list(e.get("evidence")),
                "attrs": e.get("attrs") or {},
                "updated_at": now,
            }})
        q = """
        UNWIND $rows AS row
        MERGE (n:Entity {id: row.props.id})
        SET n += row.props
        """
        with self.driver.session() as s:
            for i in range(0, len(rows), batch):
                s.run(q, rows=rows[i:i+batch])

    def upsert_relations(self, ents: List[Dict], rels: List[Dict], batch: int = 500):
        if not rels:
            return
        # (이 함수는 s/o가 id 또는 이름일 수 있음 → id 우선, 없으면 canonical로 매칭)
        id_set = {e["id"] for e in ents if e.get("id")}
        name2id = {}
        for e in ents:
            can = e.get("canonical")
            if e.get("id") and can:
                name2id[norm(can)] = e["id"]

        rows = []
        now = datetime.now(ZoneInfo("Asia/Seoul"))
        for r in rels:
            s_raw = str(r.get("s", ""))
            o_raw = str(r.get("o", ""))
            p = str(r.get("p", ""))
            p_norm = norm(r.get("p_norm", p))
            s_id = s_raw if s_raw in id_set else name2id.get(norm(s_raw))
            o_id = o_raw if o_raw in id_set else name2id.get(norm(o_raw))
            if not s_id or not o_id:
                continue
            rows.append({
                "s_id": s_id, "o_id": o_id,
                "p": p, "p_norm": p_norm,
                "evidence": uniq_list(r.get("evidence")),
                "source": uniq_list(r.get("source")),
                "updated_at": now
            })

        if not rows:
            return
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
        with self.driver.session() as s:
            for i in range(0, len(rows), batch):
                s.run(q, rows=rows[i:i+batch])

    
    def delete_entities_by_canonical_type(self, items: List[Tuple[str, str]]):
        if not items:
            return
        q = """
        UNWIND $rows AS row
        MATCH (n:Entity {canonical: row.canonical, type: row.type})
        DETACH DELETE n
        """
        rows = [{"canonical": c, "type": t} for (c, t) in items]
        with self.driver.session() as s:
            s.run(q, rows=rows)

    def delete_relations_by_triplet(self, items: List[Tuple[str, str, str]]):
        """
        items: (s_canonical, p_norm, o_canonical)
        """
        if not items:
            return
        q = """
        UNWIND $rows AS row
        MATCH (s:Entity {canonical: row.s})
              -[r:REL {p_norm: row.p_norm}]->
              (o:Entity {canonical: row.o})
        DELETE r
        """
        rows = [{"s": s, "p_norm": p, "o": o} for (s, p, o) in items]
        with self.driver.session() as s:
            s.run(q, rows=rows)

# ----------------------------
# Diff 결과 구조체
# ----------------------------
@dataclass
class EntityDiff:
    added: List[Dict]
    removed: List[Dict]
    changed: List[Tuple[Dict, Dict, Dict]]  # (old, new, changes)

@dataclass
class RelationDiff:
    added: List[Dict]
    removed: List[Dict]
    changed: List[Tuple[Dict, Dict, Dict]]  # (old, new, changes)

@dataclass
class KBDiff:
    entities: EntityDiff
    relations: RelationDiff


# ----------------------------
# Diff 계산
# ----------------------------
def index_entities_by_key(ents: List[Dict]) -> Dict[str, Dict]:
    return {entity_key(e): e for e in ents}

def index_relations_by_key(rels: List[Dict], id2canon_norm: Dict[str, str] | None = None) -> Dict[str, Dict]:
    return {rel_key(r, id2canon_norm): r for r in rels}

def id2canon_norm_map(ents: List[Dict]) -> Dict[str, str]:
    m = {}
    for e in ents:
        if e.get("id"):
            m[e["id"]] = norm(e.get("canonical",""))
    return m

def diff_sets(old_map: Dict[str, Dict], new_map: Dict[str, Dict]) -> Tuple[List[Dict], List[Dict], Set[str]]:
    old_keys = set(old_map.keys())
    new_keys = set(new_map.keys())
    added_keys = new_keys - old_keys
    removed_keys = old_keys - new_keys
    common_keys = old_keys & new_keys
    added = [new_map[k] for k in sorted(added_keys)]
    removed = [old_map[k] for k in sorted(removed_keys)]
    return added, removed, common_keys

def dict_list_diff(name: str, old_list: List[str], new_list: List[str]) -> Dict:
    o = set(old_list or [])
    n = set(new_list or [])
    add = sorted(n - o)
    rem = sorted(o - n)
    return {f"{name}_added": add, f"{name}_removed": rem} if (add or rem) else {}

def entity_change(old: Dict, new: Dict) -> Dict:
    changes = {}
    if (old.get("canonical") or "") != (new.get("canonical") or ""):
        changes["canonical"] = {"old": old.get("canonical"), "new": new.get("canonical")}
    if (old.get("type") or "") != (new.get("type") or ""):
        changes["type"] = {"old": old.get("type"), "new": new.get("type")}
    # 리스트/맵 차이
    changes |= dict_list_diff("aliases", old.get("aliases") or [], new.get("aliases") or [])
    changes |= dict_list_diff("source", old.get("source") or [], new.get("source") or [])
    changes |= dict_list_diff("evidence", old.get("evidence") or [], new.get("evidence") or [])
    # attrs는 단순 덮었을 때 변경된 key만 추려봄
    olda, newa = old.get("attrs") or {}, new.get("attrs") or {}
    attr_keys = set(olda.keys()) | set(newa.keys())
    attr_diff = {}
    for k in sorted(attr_keys):
        if json.dumps(olda.get(k), ensure_ascii=False, sort_keys=True) != json.dumps(newa.get(k), ensure_ascii=False, sort_keys=True):
            attr_diff[k] = {"old": olda.get(k), "new": newa.get(k)}
    if attr_diff:
        changes["attrs"] = attr_diff
    return changes

def relation_change(old: Dict, new: Dict) -> Dict:
    changes = {}
    # p / p_norm 변화
    if (old.get("p") or "") != (new.get("p") or ""):
        changes["p"] = {"old": old.get("p"), "new": new.get("p")}
    if norm(old.get("p_norm", old.get("p",""))) != norm(new.get("p_norm", new.get("p",""))):
        changes["p_norm"] = {"old": old.get("p_norm"), "new": new.get("p_norm")}
    changes |= dict_list_diff("evidence", old.get("evidence") or [], new.get("evidence") or [])
    changes |= dict_list_diff("source", old.get("source") or [], new.get("source") or [])
    return changes

def compute_kb_diff(
    old_ents: List[Dict], old_rels: List[Dict],
    new_ents: List[Dict], new_rels: List[Dict],
) -> KBDiff:
    
    old_e_map = index_entities_by_key(old_ents)
    new_e_map = index_entities_by_key(new_ents)

    e_added, e_removed, e_common = diff_sets(old_e_map, new_e_map)
    e_changed = []
    for k in sorted(e_common):
        ch = entity_change(old_e_map[k], new_e_map[k])
        if ch:
            e_changed.append((old_e_map[k], new_e_map[k], ch))

    old_id2canon = id2canon_norm_map(old_ents)
    new_id2canon = id2canon_norm_map(new_ents)
    old_r_map = index_relations_by_key(old_rels, old_id2canon)
    new_r_map = index_relations_by_key(new_rels, new_id2canon)

    r_added, r_removed, r_common = diff_sets(old_r_map, new_r_map)
    r_changed = []
    for k in sorted(r_common):
        ch = relation_change(old_r_map[k], new_r_map[k])
        if ch:
            r_changed.append((old_r_map[k], new_r_map[k], ch))

    return KBDiff(
        entities=EntityDiff(added=e_added, removed=e_removed, changed=e_changed),
        relations=RelationDiff(added=r_added, removed=r_removed, changed=r_changed),
    )


def print_kb_diff_summary(diff: KBDiff, limit: int = 10):
    print("=== KB Diff Summary ===")
    print(f"Entities: +{len(diff.entities.added)} / -{len(diff.entities.removed)} / ~{len(diff.entities.changed)}")
    print(f"Relations: +{len(diff.relations.added)} / -{len(diff.relations.removed)} / ~{len(diff.relations.changed)}")

    def brief_e(e): return f"{e.get('type','?')} :: {e.get('canonical','?')}"
    def brief_r(r): return f"{r.get('s','?')} -[{r.get('p_norm', r.get('p','?'))}]-> {r.get('o','?')}"

    if diff.entities.added:
        print("\n[Entities Added] (up to", limit, ")")
        for e in diff.entities.added[:limit]:
            print(" +", brief_e(e))
    if diff.entities.removed:
        print("\n[Entities Removed] (up to", limit, ")")
        for e in diff.entities.removed[:limit]:
            print(" -", brief_e(e))
    if diff.entities.changed:
        print("\n[Entities Changed] (up to", limit, ")")
        for old, new, ch in diff.entities.changed[:limit]:
            print(" ~", brief_e(old), "=>", brief_e(new), "| changes:", ch)

    if diff.relations.added:
        print("\n[Relations Added] (up to", limit, ")")
        for r in diff.relations.added[:limit]:
            print(" +", brief_r(r))
    if diff.relations.removed:
        print("\n[Relations Removed] (up to", limit, ")")
        for r in diff.relations.removed[:limit]:
            print(" -", brief_r(r))
    if diff.relations.changed:
        print("\n[Relations Changed] (up to", limit, ")")
        for old, new, ch in diff.relations.changed[:limit]:
            print(" ~", brief_r(old), " | changes:", ch)


def apply_diff_to_neo4j(
    neo: Neo4jIO,
    new_ents: List[Dict],
    new_rels: List[Dict],
    diff: KBDiff,
    delete_removed: bool = False
):
    # 추가/변경 업서트
    ents_to_upsert = diff.entities.added + [new for _, new, _ in diff.entities.changed]
    if ents_to_upsert:
        neo.upsert_entities(ents_to_upsert)
    rels_to_upsert = diff.relations.added + [new for _, new, _ in diff.relations.changed]
    if rels_to_upsert:
        neo.upsert_relations(new_ents, rels_to_upsert)

    # 삭제(선택)
    if delete_removed:
        # 엔터티 삭제는 관계 참조가 있을 수 있으니 신중히.
        ent_del_items = [(e.get("canonical",""), e.get("type","")) for e in diff.entities.removed]
        if ent_del_items:
            neo.delete_entities_by_canonical_type(ent_del_items)

        rel_del_items = []
        for r in diff.relations.removed:
            s = norm(r.get("s",""))
            p = norm(r.get("p_norm", r.get("p","")))
            o = norm(r.get("o",""))
            rel_del_items.append((s, p, o))
        if rel_del_items:
            neo.delete_relations_by_triplet(rel_del_items)
