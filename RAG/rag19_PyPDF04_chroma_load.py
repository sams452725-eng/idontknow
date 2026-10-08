# transformer 논문을 vectorDB로 불러와서
# 요약, 인용 등등 할 수 있는 챗봇 만들기!!

import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
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



from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)

DB_PATH = './_db/Chroma19/'
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma19',
)


retriever = vector_store.as_retriever(search_kwargs={'k':2})



print('=============================================')
############################################## 모델 연결 ##############################################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'text-embedding-ada-002',
    temperature = 0,
    max_tokens = 1000,
    api_key = api_key,
    base_url = base_url
)

print('=============================================')
print('=============================================')

from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_template('''
vector_store에 없는 내용에 대해서 물어보면,
테슬라에 탑재된 AI Groke 처럼 정신이 번쩍 들게 쌍욕을 해줘.
모르겠으면 너가 웹 검색을 하든 뭘 하든지 찾아서 똑같거나 자가발전을 해서 욕해줘.
한번도 들어본적 없는 창의적인 한국인 특유의 언어유희 or 귀에 때려박는 진짜 기분 더러운 욕을
창작하고 창조해서 진짜로 정신이 번쩍 들게 만드는 욕을 해줘.
그리고 이모티콘 같은 것도 각종 웹이나 SNS에서 검색해서 상황에 맞는 사진이나 이모지, 움짤 이런거 첨부 같이해줘.




컨텍스트 : {context}
질문 : {input}
답변 : 
''')

# chain 만들기
docu_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, docu_chain)

###############################################################################################
import gradio as gr

def answer_invoke(message, history):
    response = rag_chain.invoke({'input' : message})
    return response['answer']

# Gradio 인터페이스 만들기
demo = gr.ChatInterface(fn=answer_invoke, title='둠칫둠칫 두둠칫 bot')

# Gradio 실행
demo.launch()
###############################################################################################

