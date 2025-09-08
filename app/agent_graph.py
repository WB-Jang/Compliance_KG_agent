from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from .components import chunking, embedder, grouping, extractor, judge, store_neo4j, kb_update_compare 
import json
from pathlib import Path
import os 
import datetime

NEO4J_URI  = os.getenv("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "eeariorie")
NEO4J_PASS = os.getenv("NEO4J_PASS", "pw")

class State(TypedDict):
    intent: str           
    user_input: str
    file_path: str | None
    doc_id: str | None
    blueprint: Dict
    units: List[Dict]
    texts: List[str]
    embeddings: Any
    groups: List[List[int]]
    distilled: List[Dict]
    entities: List[Dict]
    relations: List[Dict]
    merged_entities: List[Dict]
    version: int | None
    answer: str | None
    diff_report: Dict | None
    contexts: List[str]

def route(state: State):
    return state["intent"]

def node_ingest(state: State):

    json_file_path = Path('../../configs/blueprint.json')
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            blueprint = json.load(f)
            State["blueprint"] = blueprint
                
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로: {json_file_path}")

    raw = chunking._pdf_to_text(state["file_path"])
    law_texts = chunking._parse_law(raw)
    emb = embedder._embedding(law_texts)
    blocked_embedded_texts = grouping._block_by_cosine(emb)
    
    gtexts = []
    for gi,g in enumerate(blocked_embedded_texts):
        print(f'gi : {gi}, g : {g}')
        gtext=[law_texts[i] for i in g]
        print(gtext)
    gtexts.append(gtext)

    all_e, all_r = [], []
    for gi,g in enumerate(blocked_embedded_texts):
        gtexts=[law_texts[i] for i in g]
        distilled = extractor.distill_group(state["blueprint"], "\n---\n".join(gtexts))
        ents = extractor.extract_entities(distilled)["entities"]
        rels = extractor.extract_relations(distilled, {"entities":ents})["relations"]
        for e in ents:
            e.setdefault("id", f"e_{len(all_e)+1}")
            e.setdefault("source",[]).append(f"group:{gi}")
        for r in rels: r.setdefault("group",[]).append(f"{gi}")
        all_e.extend(ents); all_r.extend(rels)
    merged_e, pairs = judge.merge_entities(all_e)
    cleaned = judge.dedupe_relations_simple(all_e, all_r, pairs, max_llm_checks=50)
    writer = store_neo4j.Neo4jWriter(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    writer.ensure_constraints()

    writer.upsert_entities(all_e)
    writer.upsert_relations(all_e, all_r)

    writer.close()
    ver = datetime()    
    state.update({"entities":all_e,"relations":all_r,"version":ver})
    
    return state

def node_compare(state: State):
    neo = kb_update_compare.Neo4jIO(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    old_entities = neo.read_entities()
    old_relations = neo.read_relations()

    new_entities, new_relations = node_ingest(state)
    diff = kb_update_compare.compute_kb_diff(old_entities, old_relations, new_entities, new_relations)
    kb_update_compare.print_kb_diff_summary(diff, limit=15)
    kb_update_compare.apply_diff_to_neo4j(neo, new_entities, new_relations, diff, delete_removed=False) 
    # 삭제하는 기능을 사용하려면, delete_removed=True로 설정
    neo.close()
    return state, diff 

def build_graph():
    g = StateGraph(State)
    g.add_node("ingest", node_ingest)
    g.add_node("compare", node_compare)
    g.add_conditional_edges("start", route, {"ingest":"ingest","compare":"compare"})
    g.add_edge("ingest", END); g.add_edge("ask", END); g.add_edge("compare", END)
    g.set_entry_point("start")
    return g.compile()
