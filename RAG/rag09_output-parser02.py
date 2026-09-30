# LCEL = Langchain Expression Language
# chain = prompt | model | output_parser

# parser : 분석하다. 답변을 다듬다.

# 문자에서 '''는 문장 주석이 아니라 범위 내에 전체 내용을 입력한다.

template = '''
당신은 영어를 가르치는 10년차 영어 선생님입니다.
주어진 상황에 맞는 영어 회화에 맞는 영어회화를 작성해 주세요.
양식은 [FORMAT]을 참고하여 작성해주세요.

# 상황 :
{question}


# FORMAT :
-영어회화 : 
-한글번역 :
'''

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = PromptTemplate.from_template(template=template)

model = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    api_key=api_key,
    base_url=base_url,
)

from langchain_core.output_parsers import StrOutputParser     #output_parser를 Str(문자)로 빼주겠다는 의미.
output_parser = StrOutputParser()

chain = prompt | model | output_parser

input = {'question' : '저는 부산에서 물밀면을 먹고싶어요.'}

response = chain.invoke(input)
print(response)    # 위에서 output_parser를 Str로 뺐기 때문에 더이상 .content가 필요가 없다. 왜? 어차피 문자로 반환될거니까!!

