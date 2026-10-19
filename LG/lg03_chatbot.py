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

#################### 1. State 정의 ####################
class State(TypedDict):
    messages : Annotated[list, add_messages]
    aaa : str
    bbb : str
    ccc : int

# 위에 Class State는 한마디로 아래의 형태와 같다!!
# state = {
#     'messages' : [],
#     'aaa' : 'abc',
#     'bbb' : 'bcd',
#     'ccc' : 'cde',
# }

#################### 2. node 정의 ####################
def chatbot(state : State):
    return {'messages' : [llm.invoke(state['messages'])]}


#################### 3. graph 구성 ####################
graph_01 = StateGraph(State)

graph_01.add_node('chatbot', chatbot)    # lang_graph에 chatbot이란 node를 엮을거다.(아직 엮은건 아님)


#################### 4. edge 추가 ####################
graph_01.add_edge(START, 'chatbot')

graph_01.add_edge('chatbot', END)


#################### 5. graph compile ####################
graph = graph_01.compile()

#################### 6. graph 실행 ####################
query = '삼성전자의 주가를 예측해줘'

response = graph.invoke({'messages' : [('user', query)]})
# print(response['messages'][0])            # HumanMessage
# print(response['messages'][1])            # AIMessage
# print(response['messages'][1].content)
# print(response['messages'][0].type)       # human
# print(response['messages'][1].type)       # ai