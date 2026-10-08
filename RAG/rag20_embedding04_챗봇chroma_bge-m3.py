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


from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings( 
    model_name='BAAI/bge-m3',               # 임베딩 백터의 차원 : 1024
    model_kwargs={
        'device' : 'cpu',
        'local_files_only' : True,
    }
)


DB_PATH = './_db/Chroma_bge/'
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma_bge',
)

query = '삼성전자의 창업자는 누구인가?'
result = vector_store.similarity_search(query)
retriever = vector_store.as_retriever(search_kwargs={'k':2})



print('=============================================')
############################################## 모델 연결 ##############################################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'claude-fable-5',
    temperature = 0,          # 대답의 창의성 '1' = 창의적으로, '0' = 있는 그대로
    max_tokens = 1000,        # 갈수록 토큰 사용량이 많아지니까 초대의 토큰을 내가 설정한다.
    api_key = api_key,
    base_url = base_url
)

# response = model.invoke('삼성전자의 창업주는 누구인가?')
# print('model의 답변 :', response.content)
print('=============================================')
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
너가 진짜 사람인것처럼 생각해서 직접 답변해주고, 그래도 어렵고 대답을 못하겠으면
'주어진 정보로는 답변할 수 없습니다.'라고 말씀해주세요.


컨텍스트 : {context}
질문 : {input}
답변 : 
''')

# chain 만들기
docu_chain = create_stuff_documents_chain(model, prompt)      # prompt | model -----> 구조를 지금 짜고 있다.
rag_chain = create_retrieval_chain(retriever, docu_chain)     # 검색 | docu_chain ------> 검색과 vectorDB를 연결하겠다.
# create_stuff_documents_chain와 create_retrieval_chain 두개는 세트로 늘 같이 나온다.
# prompt와 model을 연결하고 retriever와 docu_chain을 연결해서 전체를 하나로 묶는다.


'''
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


###############################################################################################
import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({'input' : message})
    return response['answer']

# Gradio 인터페이스 만들기
demo = gr.ChatInterface(fn=answer_invoke, title='둠칫둠칫 봇치 더 락')

# Gradio 실행
demo.launch()
# demo.launch(share=True) ------------> public 링크도 생성해서 타인이 접속 가능하게 만들어준다.
###############################################################################################
# 이게 베스트는 아니다. 다만, 직관적이고 빠르게 만들어지기 때문에 쓴다.

