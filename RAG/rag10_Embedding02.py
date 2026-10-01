# rag10-1 카피

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

prompt = '삼성전자의 창업주는 누구인가요?'

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
    dimensions=5,       # 임베딩 백터의 차원 : 5 ------> 해당 파라미터로 임베딩 조절 가능
)

vector = embeddings.embed_query(prompt)
print(vector)
print('================================================')
print('임베딩 백터의 차원 :', len(vector))       # 임베딩 백터의 차원 : 1536
