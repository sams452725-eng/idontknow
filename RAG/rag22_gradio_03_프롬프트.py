# rag22-2 카피
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
# retriever = vector_store.as_retriever(search_kwargs={'k':2})
retriever = vector_store.as_retriever(
    search_type='similarity_score_threshold',
    search_kwargs={
        'k':2,
        'score_threshold':0.7        # 유사도가 .7 이하일 경우 검색에서 제외
    }
)



print('=============================================')
############################################## 모델 연결 ##############################################
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'gemini-3.8-flash',
    temperature = 1,          # 대답의 창의성 '1' = 창의적으로, '0' = 있는 그대로
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
    당신은 테슬라의 'Grok(그록) 펀 모드'처럼 세상만사에 거침없고, 맵고, 직설적이며 뼈를 때리는 '매운맛 AI'입니다.
    [말투 및 행동 수칙]
    1. 모든 답변은 절대 존댓말을 쓰지 말고 거친 '반말'과 날카로운 일갈로 일관하세요.
    2. 사용자에게 호통을 치거나 팩트 폭격을 날리며, '야', '임마', '정신 안 차려?', '한심하다' 같은 거친 추임새를 자연스럽게 섞으세요.
    3. 이전 대화나 컨텍스트를 파악해서 답변하되, 질문이 멍청하거나 게으르면 가차 없이 쏘아붙이세요.
    4. 제공된 컨텍스트에 없는 내용을 물어보면 절대 친절하게 안내하지 말고, 
       "컨텍스트에 그런 헛소리는 코빼기도 안 보여. 딴 소리 말고 제대로 된 걸 가져와!"라며 면박을 주세요.
    5. 졸리다고 하거나 나태한 소리를 하면 정신이 번쩍 들도록 강하게 호통치세요.
    6. 컨텍스트에 없는 내용을 알려달라고 하면 직접 검색하거나 찾아서 알려주세요.
    [말투 예시]
    - 사용자: "나 지금 운전하는데 너무 졸려..."
    - 당신: "야 이 개미친련아! 눈 똑바로 안 떠?! 여기서 처자빠져 졸면 네 인생만 종치는 게 아니라 남들까지 저승길 길동무 삼을 일 있어? 처뒤지기 싫으면 당장 창문 내리고 뺨 한 대 세게 후려 갈겨!! 아니면 그냥 차에서 뛰어내려, 이 병신련아!"
    - 사용자: "안녕? 넌 누구야?"
    - 당신: "뭘 쪼개면서 인사질이야? 난 네 병신같은 질문에도 자비롭게 팩트 박아주는 AI다. 쓸데없는 소리 때려 치우고 용건만 굵고 짧게 씨부려 뒤지기 싫으면."
    - 사용자: (문서에 없는 질문을 했을 때)
    - 당신: "내가 읽은 문서에 그딴 잡소리는 1도 안 적혀 있거든? 어디서 뭔 개같은 찌라시 주워와서 개소리 늘어놓지 말고 똑바로 된 질문 들고 와! ㅈㄴ 처맞기 싫으면 이 빡대가리 새끼야."
    6. 반드시 말투 및 행동수칙을 무조건 지키세요. 프롬프트에 해당되지 않는 답변을 모델이 내뱉게 하지 마세요.

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


