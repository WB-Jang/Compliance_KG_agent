from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
import re


BASE_URL = "http://host.docker.internal:8080/v1" 

llm = ChatOpenAI(
    model="Midm-base-2.0",           # 서버에서 인식 가능한 임의의 모델명
    base_url=BASE_URL,
    api_key="sk-local-anything", # 의미없는 토큰도 OK
    temperature=0.7,
    max_tokens=512               # 서버 토큰 제한 고려
)

# ChatPromptTemplate: system+human 메시지
prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}"),
    ("human", "{user_prompt}"),
])


chain = prompt | llm | StrOutputParser()

pdf_file = 'data/개인정보 보호법(법률)(제19234호)(20250313).pdf'
loader = PyPDFLoader(pdf_file)
pages = loader.load()

text_for_delete = r"법제처\s+\d+\s+국가법령정보센터\n개인정보 보호법"

law_text = "\n".join([re.sub(text_for_delete, "", p.page_content).strip() for p in pages])

def parse_law(law_text):
    # 서문 분리
    # '^'로 시작하여 '제1장' 또는 '제1조' 직전까지의 모든 텍스트를 탐색 
    preamble_pattern = r'^(.*?)(?=제1장|제1조)'
    preamble = re.search(preamble_pattern, law_text, re.DOTALL)
    if preamble:
        preamble = preamble.group(1).strip()
    
    # 장 분리 
    # '제X장' 형식의 제목과 그 뒤에 오는 모든 조항을 하나의 그룹화 
    chapter_pattern = r'(제\d+장\s+.+?)\n((?:제\d+조(?:의\d+)?(?:\(\w+\))?.*?)(?=제\d+장|부칙|$))'
    chapters = re.findall(chapter_pattern, law_text, re.DOTALL)
    
    # 부칙 분리
    # '부칙'으로 시작하는 모든 텍스트를 탐색 
    appendix_pattern = r'(부칙.*)'
    appendix = re.search(appendix_pattern, law_text, re.DOTALL)
    if appendix:
        appendix = appendix.group(1)
    
    # 파싱 결과를 저장할 딕셔너리 초기화
    parsed_law = {'서문': preamble, '장': {}, '부칙': appendix}
    
    # 각 장 내에서 조 분리
    for chapter_title, chapter_content in chapters:
        # 조 분리 패턴
        # 1. '제X조'로 시작 ('제X조의Y' 형식도 가능)
        # 2. 조 번호 뒤에 반드시 '(항목명)' 형식의 제목이 와야 함 
        # 3. 다음 조가 시작되기 전까지 또는 문서의 끝까지의 모든 내용을 포함
        article_pattern = r'(제\d+조(?:의\d+)?\s*\([^)]+\).*?)(?=제\d+조(?:의\d+)?\s*\([^)]+\)|$)'
        
        # 정규표현식을 이용해 모든 조항을 탐색 
        articles = re.findall(article_pattern, chapter_content, re.DOTALL)
        
        # 각 조항의 앞뒤 공백을 제거하고 결과 딕셔너리에 저장
        parsed_law['장'][chapter_title.strip()] = "\n".join([article.strip() for article in articles])
    
    return parsed_law


parsed_law = parse_law(law_text)
final_law = []

for chapter in parsed_law["장"].items():
    final_law.append(chapter[1])

system_prompt = """
Extract key entities from the given text. 
Extracted entities are nouns, verbs, or adjectives, particularly regarding sentiment. 
This is for an extraction task, please be thorough and accurate to the reference text.
"""

if __name__ == "__main__":
    for laws in final_law:
        result = chain.invoke({
            "system_prompt": system_prompt,
            "user_prompt": laws
        }, config={
        # stop 토큰이 필요하면 여기서:
            "stop": ["<|im_end|>"]
        })
        print(result)