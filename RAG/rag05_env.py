from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()          # 작업그룹 내에 .env를 찾아서 가져온다.
# .env를 쓰기 위해서는 원칙적으로는 위의 import와 load를 해야하지만 vs_code는 자체 시스템에서 .env의 키를 알아서 인식해준다. 리눅스나 cmd에서 직접 작업할땐 반드시 있어야 한다.

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    # openai_api_key = openai_api_key,
)

response = llm.invoke('너의 이름은?')      
print(response.content)    
