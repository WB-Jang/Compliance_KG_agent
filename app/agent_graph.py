from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from .components import chunking, embedder, grouping, extract, judge, store, diff as diffmod

class State(TypedDict):
    intent: str           # "ingest" | "ask" | "compare"
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
    raw = chunking.pdf_to_text(state["file_path"])
    blocks = chunking.split_articles(raw); units = chunking.make_units(blocks)
    texts=[u["text"] for u in units]
    emb = embedder.Embedder(); X = emb.encode(texts)
    groups = grouping.block_by_cosine(X)
    # 그룹별 증류→추출
    all_e, all_r = [], []
    for gi,g in enumerate(groups):
        gtexts=[texts[i] for i in g]
        distilled = extract.distill_group(state["blueprint"], gtexts)
        ents = extract.extract_entities(distilled)["entities"]
        rels = extract.extract_relations(distilled, {"entities":ents})["relations"]
        for e in ents:
            e.setdefault("id", f"e_{len(all_e)+1}")
            e.setdefault("source",[]).append(f"group:{gi}")
        for r in rels: r.setdefault("evidence",[]).append(f"group:{gi}")
        all_e.extend(ents); all_r.extend(rels)
    merged = judge.merge_entities(all_e)
    # 버전 스냅샷
    prev_v, prev = store.load_latest_version(state["doc_id"])
    ver = (prev_v or 0) + 1
    store.save_snapshot(state["doc_id"], ver, merged, all_r, {"units":len(units)})
    store.build_vector_index(state["doc_id"], [f"u{i}" for i in range(len(texts))], texts, X)
    state.update({"units":units,"texts":texts,"embeddings":X,"groups":groups,
                  "entities":all_e,"relations":all_r,"merged_entities":merged,"version":ver})
    return state

def node_ask(state: State):
    # 간단: 최신 스냅샷 불러와 답변만 (필요시 hybrid retrieve 확장)
    _, snap = store.load_latest_version(state["doc_id"])
    ents, rels = (snap or {}).get("entities",[]), (snap or {}).get("relations",[])
    sys = "You are a compliance assistant grounded by entities/relations. Cite evidence ids when relevant."
    context = {"entities": ents[:200], "relations": rels[:200]}
    user = f"{state['user_input']}\nUse the provided KG context strictly."
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    ans = llm.invoke([("system",sys),("user",user)]).content
    state["answer"]=ans; state["contexts"]=[str(context)]
    return state

def node_compare(state: State):
    # 두 버전 번호가 user_input에 들어있다고 가정: "compare v1..v3 for DOC:xxx"
    import re, json, os
    m=re.search(r"v(\d+)\D+v(\d+)", state["user_input"])
    if not m: 
        state["diff_report"]={"error":"버전 형식 예: 'compare v1..v3'"}; return state
    v1, v2 = int(m.group(1)), int(m.group(2))
    base=f"data/graphs/{state['doc_id']}"
    j1=json.loads(open(f"{base}/version_{v1}.json",encoding="utf-8").read())
    j2=json.loads(open(f"{base}/version_{v2}.json",encoding="utf-8").read())
    add_rel, rm_rel = diffmod.diff_relations(j1["relations"], j2["relations"])
    add_ent, rm_ent = diffmod.diff_entities(j1["entities"], j2["entities"])
    id2art = {}  # (선택) id→조문 맵
    impacted = diffmod.impacted_articles(add_rel, rm_rel, id2art)
    state["diff_report"] = {
        "added_relations": list(add_rel)[:100],
        "removed_relations": list(rm_rel)[:100],
        "added_entities": list(add_ent)[:100],
        "removed_entities": list(rm_ent)[:100],
        "impacted_articles": sorted(list(impacted))
    }
    return state

def build_graph():
    g = StateGraph(State)
    g.add_node("ingest", node_ingest)
    g.add_node("ask", node_ask)
    g.add_node("compare", node_compare)
    g.add_conditional_edges("start", route, {"ingest":"ingest","ask":"ask","compare":"compare"})
    g.add_edge("ingest", END); g.add_edge("ask", END); g.add_edge("compare", END)
    g.set_entry_point("start")
    return g.compile()
