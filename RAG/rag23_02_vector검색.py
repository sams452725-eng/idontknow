# rag23-1 카피

from langchain_core.documents import Document

texts = [
    Document(page_content="삼성전자는 반도체를 생산하는 기업이다."),
    Document(page_content="삼성전자는 갤럭시 스마트폰을 판매한다."),
    Document(page_content="SK하이닉스는 메모리 반도체를 생산한다."),
    Document(page_content="엔비디아는 인공지능 연산용 GPU를 설계한다."),
    Document(page_content="애플은 아이폰을 개발하는 기업이다."),
    Document(page_content="테슬라는 전기자동차를 제조한다."),
    Document(page_content="삼성전자는 AI 반도체 기술을 연구한다."),
    Document(page_content="반도체는 컴퓨터의 핵심 부품이다."),
]

query = '삼성전자의 반도체 사업'

########################## BM25 ##########################
# BM25는 키워드 기반 검색.(BM = Best Match)  ----------> 단어의 중복이나 빈도수 등에 따른 검색(유사도 검색과는 다른 개념의 검색 방법이다.)
# from langchain_community.retrievers import BM25Retriever

# bm25_retriever = BM25Retriever.from_documents(texts)
# bm25_retriever.k = 3

# bm25_docs = bm25_retriever.invoke(query)

# print('='*25,'BM25 검색 결과','='*25)
# for doc in bm25_docs:
#     print(doc.page_content)


##########################################################
##########################################################
######################## 벡터검색 ########################
import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore


from dotenv import load_dotenv
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'


#03 임베딩.
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)


DB_PATH = './_db/Chroma12/'
vector_store = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma12',
)

vector_retriever = vector_store.as_retriever(
    search_kwargs = {'k' : 3}
)
vector_docs = vector_retriever.invoke(query)


print('='*25,'vectorDB 검색 결과','='*25)
for doc in vector_docs:
    print(doc.page_content)