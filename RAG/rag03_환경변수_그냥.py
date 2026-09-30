from langchain_openai import ChatOpenAI
import os
# os.environ['OPENAI_API_KEY'] = 'sk-proj-YOUR_API_KEY_HERE'
# 위의 코드를 직접 쓰지 않아도 작동이 되는 이유는 '검색 - 시스템 환경변수 편집' - 환경변수(N) - 새로만들기(N) - 변수이름, 변수값 입력'에서 os 자체에 입력했기 때문이다.

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    # openai_api_key = openai_api_key,
)

response = llm.invoke('안녕하세요.')      
print(response.content)    
