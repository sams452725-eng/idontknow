# rag12-4 카피

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

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

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma12',
)


print(f'벡터 저장소에 저장된 문서 수 : :{vector_store._collection.count()}', )  # 벡터 저장소에 저장된 문서 수 : :260

query = '삼성전자의 창업자는 누구인가?'
result = vector_store.similarity_search(query)

print(f'검색 결과의 길이 : {len(result)}')    # 검색 결과의 길이 : 4

####################################### Retrievers #######################################
#######################################   검색기   #######################################
retriever = vector_store.as_retriever(search_kwargs={'k':2})
print(retriever)
aaa = retriever.invoke(query)
print(f'검색된 관련 문서 수 : {len(aaa)}')   # 검색된 관련 문서 수 : 2
print(f'첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}')    # 첫번째 관련 문서 내용 미리보기 : 삼성전자 사업 전망



print('=============================================')
############################################## 모델 연결 ##############################################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-5-nano',
    temperature = 0,          # 대답의 창의성 '1' = 창의적으로, '0' = 있는 그대로
    max_tokens = 1000,        # 갈수록 토큰 사용량이 많아지니까 초대의 토큰을 내가 설정한다.
    api_key = api_key,
    base_url = base_url
)

response = model.invoke('삼성전자의 창업주는 누구인가?')
print('model의 답변 :', response.content)
print('=============================================')

query_with_context = f'''
    {aaa[0].page_content}\n\n
    위 내용에 근거하여 다음 질문에 답변하세요.\n\n{query}
'''

response = model.invoke(query_with_context)
print('model의 응답 :', response.content)