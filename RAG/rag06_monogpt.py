from langchain_openai import ChatOpenAI
import os

from dotenv import load_dotenv
load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()      # .strip()은 줄바꿈이나 공백을 인정하지 않는 기능. 오타방지용
base_url = 'https://monogpt.kr/api/monorouter/v1'       # major가 아닌 일반 여러 api를 제공하는 회사들은 api_key와base_url을 같이 제공해줘야 한다.

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

response = llm.invoke('너의 이름은?')      
print(response.content)    
