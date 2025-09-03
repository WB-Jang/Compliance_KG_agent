from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import re

# 임베딩 모델을 사용하고 은행에서 사용한 버전을 그대로 사용할 것
# llama server 사용 방식은 llama.cpp 폴더 안에 가이드 파일 참조

BASE_URL = "http://host.docker.internal:8080/v1" 

llm = ChatOpenAI(
    model="bge-m3",           # 서버에서 인식 가능한 임의의 모델명
    base_url=BASE_URL,
    api_key="sk-local-anything", # 의미없는 토큰도 OK
    temperature=0.7,
    max_tokens=512               # 서버 토큰 제한 고려
)

# 임베딩 결과를 vstack을 사용하여 ndarray 형식으로 return할 것
