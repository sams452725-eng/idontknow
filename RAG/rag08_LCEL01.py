# LCEL = Langchain Expression Language
# chain = prompt | model | output_parser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = PromptTemplate.from_template('{topic}에 대해 쉽게 설명해주세요.')

model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

chain = prompt | model      # python에서 '|' 연산자는 'or'랑 똑같은 의미이다.-----> 하지만 langchain에서는 prompt는 model로 하라는 덮어쓰기 개념의 overriding으로 재정의 했다.
input = {'topic' : '양자컴퓨터 학습 원리'}

response = chain.invoke(input)
print(response.content)