# rag21 카피
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
    model_name='Qwen/Qwen3-Embedding-0.6B', 
    model_kwargs={
        'device' : 'cpu',
        'local_files_only' : True,
    }
)

DB_PATH = './_db/FAISS20-qwen3/'
vector_store = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_qwen3',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
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


from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
prompt = ChatPromptTemplate.from_messages([
    ('system', '''
    다음 컨텍스트를 바탕으로 질문에 답변해 주세요. 컨텍스트 관련 정보가 없다면,
    '주어진 정보로는 답변할 수 없습니다.'라고 말씀해주세요.


    컨텍스트 : {context}
'''),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{input}')])


# chain 만들기
docu_chain = create_stuff_documents_chain(model, prompt)
rag_chain = create_retrieval_chain(retriever, docu_chain)


###############################################################################################
import gradio as gr


def answer_invoke(message, history):
    chat_history = []
    for item in history[-10:]:
        content = item['content']

        if item['role'] == 'user':
            chat_history.append(HumanMessage(content=content))
        elif item['role'] == 'assistant':
            chat_history.append(AIMessage(content=content))

    response = rag_chain.invoke({
        'input' : message,
        'chat_history' : chat_history
    })

    # 터미널 상으로 실제 검색된 문서 2개 확인.
    for i, doc in enumerate(response['context'], start=1):
        print(f'\n검색 문서{i}')
        print('출처 :', doc.metadata.get('source'))
        print('내용 :', doc.page_content)
        print('='*25)





    return response['answer']

# Gradio 인터페이스 만들기
demo = gr.ChatInterface(fn=answer_invoke, title='둠칫둠칫 봇치 더 락')

# Gradio 실행
demo.launch()


