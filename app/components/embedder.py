from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import requests
import json, os 
import numpy as np 
import re

# 임베딩 모델을 사용하고 은행에서 사용한 버전을 그대로 사용할 것
# llama server 사용 방식은 llama.cpp 폴더 안에 가이드 파일 참조

BASE_URL = "http://host.docker.internal:8080/v1" 

llm = ChatOpenAI(
    model="bge-m3",           # 서버에서 인식 가능한 임의의 모델명
    base_url=BASE_URL,
    api_key="sk-local-anything", # 의미없는 토큰도 OK
    temperature=0.7,
    max_tokens=4096               # 서버 토큰 제한 고려
)

def _embedding(texts):
    """
    llama-server의 v1/embeddings 엔드 포인트를 사용해 bge-m3.gguf 임베딩 진행
    반환 결과 : (N,D) float32 numpy array
    N = chunk나 sentence 개수
    D = Vector의 차원(bge-m3는 1,024로 고정)
    """
    out = []
    for t in texts:
        vecs = None
        try:
            r= requests.post(f"{BASE_URL}/embeddings", json={"model": "bge-m3", "input": t}, timeout=45)
            if r.ok and "data" in r.json():
                vecs = r.json()["data"][0]["embedding"]
                v = np.array(vecs, dtype=np.float32)
                v /= (np.linalg.norm(v)+1e-12)
                out.append(v)
        except Exception:
            pass
    if vecs is None:
        raise RuntimeError(f"embedding API 실패 : {t}")

    return np.vstack(out).astype("float32")
