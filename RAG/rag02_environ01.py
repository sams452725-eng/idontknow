from langchain_openai import ChatOpenAI
import os
os.environ['OPENAI_API_KEY'] = 'sk-proj-YOUR_API_KEY_HERE'
# 위의 코드는 내 컴퓨터의 os에 환경변수에 저 key값을 직접 넣겠다라는 의미. 단, 일회성이기 때문에 현재 스크립트에서만 사용이 가능하다.(이또한 rag01처럼 노출의 위험성 때문에 쓰지 않는다.)
llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    # openai_api_key = openai_api_key,       # os에 입력을 해놨기 때문에 해당 코드는 없어도 된다.
)

response = llm.invoke('안녕하세요.')      
print(response.content)    
