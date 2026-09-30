# LCEL = Langchain Expression Language
# chain = prompt | model | output_parser

# parser : 분석하다. 답변을 다듬다.

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

from langchain_core.output_parsers import StrOutputParser     #output_parser를 Str(문자)로 빼주겠다는 의미.
output_parser = StrOutputParser()

chain = prompt | model | output_parser

input = {'topic' : '양자컴퓨터 학습 원리'}

response = chain.invoke(input)
print(response)    # 위에서 output_parser를 Str로 뺐기 때문에 더이상 .content가 필요가 없다. 왜? 어차피 문자로 반환될거니까!!
