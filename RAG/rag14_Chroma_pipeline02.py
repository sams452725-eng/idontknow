# rag13-1 카피

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


# print(f'벡터 저장소에 저장된 문서 수 : :{vector_store._collection.count()}', )  # 벡터 저장소에 저장된 문서 수 : :260

query = '삼성전자의 창업자는 누구인가?'
result = vector_store.similarity_search(query)

# print(f'검색 결과의 길이 : {len(result)}')    # 검색 결과의 길이 : 4

####################################### Retrievers #######################################
#######################################   검색기   #######################################
retriever = vector_store.as_retriever(search_kwargs={'k':2})
# print(retriever)
# aaa = retriever.invoke(query)
# print(f'검색된 관련 문서 수 : {len(aaa)}')   # 검색된 관련 문서 수 : 2
# print(f'첫번째 관련 문서 내용 미리보기 : {aaa[0].page_content[:50]}')    # 첫번째 관련 문서 내용 미리보기 : 삼성전자 사업 전망



print('=============================================')
############################################## 모델 연결 ##############################################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gpt-6-luna',
    temperature = 0,          # 대답의 창의성 '1' = 창의적으로, '0' = 있는 그대로
    max_tokens = 1000,        # 갈수록 토큰 사용량이 많아지니까 초대의 토큰을 내가 설정한다.
    api_key = api_key,
    base_url = base_url
)

# response = model.invoke('삼성전자의 창업주는 누구인가?')
# print('model의 답변 :', response.content)
print('=============================================')

# query_with_context = f'''
#     {aaa[0].page_content}\n\n
#     위 내용에 근거하여 다음 질문에 답변하세요.\n\n{query}
# '''

# response = model.invoke(query_with_context)
# print('model의 응답 :', response.content)

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template('''
다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
'주어진 정보로는 답변할 수 없습니다.'라고 말씀해주세요.

컨텍스트 : {context}
질문 : {input}
답변 : 
''')

# chain 만들기
docu_chain = create_stuff_documents_chain(model, prompt)      # prompt | model -----> 구조를 지금 짜고 있다.
rag_chain = create_retrieval_chain(retriever, docu_chain)     # 검색 | docu_chain ------> 검색과 vectorDB를 연결하겠다.


#  체인 실행
query = '삼성전자의 창업주는 누구인가요?'
response = rag_chain.invoke({'input' : query})

print(response)
print('====================== keys() =======================')
print(response.keys())
# dict_keys(['input', 'context', 'answer'])
print('===================== context ========================')
print(response['context'][0].page_content)
# 삼성전자 사업 전망

# 삼성전자는 메모리 반도체, 파운드리, 스마트폰, 디스플레이와 가전 사업을 운영하는 종합 전자기업이다.
# 여러 사업을 보유한 구조는 특정 시장의 부진을 다른 사업이 일부 보완할 수 있다는 장점이 있다.
# 다만 각 사업의 성장 요인과 위험이 다르므로, 회사의 전망을 살필 때에는 사업별 흐름을 구분하는 편이 정확하다.
print('======================= answer ======================')
print(response['answer'])
# 주어진 정보로는 답변할 수 없습니다. -----> why? 위에 prompt = ChatPromptTemplate.from_template에서 지정해놨기 때문이다.















'''
5.6 luna 걸린시간 :  7.62 초
6 luna 걸린시간 :  7.17 초
5.6 terra 걸린시간 :  9.75 초
5.6 sol 걸린시간 :  10.83 초
6 sol 걸린시간 :  7.42 초
6.1 sole 걸린시간 :  9.8 초
6 astra 걸린시간 :  10.07 초
'''