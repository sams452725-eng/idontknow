from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

llm = ChatOpenAI(
    model_name=' gpt-6-astra',
    temperature=0,
    max_tokens=1000,
    api_key=api_key,
    base_url=base_url,
)

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph import add_messages
from langchain_core.messages import AIMessage,HumanMessage

#################### 1. State 정의 ####################
class State(TypedDict):
    messages : Annotated[list, add_messages]
    aaa : str
    bbb : str
    ccc : int



#################### 2. node 정의 ####################
# (state : State)는 state=State()를 받아들이라는 말이다
def prepare(state : State):
    return{'messages' :
           [AIMessage(content='질문을 접수했습니다!!')]}


def chatbot(state : State):
    return {'messages' : [llm.invoke(state['messages'])]}


def finish(state : State):
    return{'messages' :
               [AIMessage(content='답변처리가 완료되었습니다.')]}




#################### 3. graph 구성 ####################
graph_01 = StateGraph(State)

graph_01.add_node('prepare', prepare)
graph_01.add_node('chatbot', chatbot)
graph_01.add_node('finish', finish)


#################### 4. edge 추가 ####################
graph_01.add_edge(START, 'prepare')
graph_01.add_edge('prepare', 'chatbot')
graph_01.add_edge('chatbot', 'finish')
graph_01.add_edge('finish', END)


#################### 5. graph compile ####################
graph = graph_01.compile()

#################### 6. graph 실행 ####################
query = '삼성전자의 주가를 예측해줘'

response = graph.invoke({'messages' : [('user', query)]})

#################### 7. 결과 출력 ####################
for i, msg in enumerate(response['messages']):
    print(f'[{i}] {msg.type} : {msg.content}')