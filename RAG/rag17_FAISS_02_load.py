# rag17-1 카피

import os
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_chroma import Chroma

# pip install faiss-cpu
import faiss     # faiss 자체 import             faiss와 langchain FAISS는 살짝의 기능차이가 있다.
from langchain_community.vectorstores import FAISS      # langchain에서 지원하는 faiss를 import
from langchain_community.docstore.in_memory import InMemoryDocstore


from dotenv import load_dotenv
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'

'''
#01 데이터를 불러온다.
path = './_data/rag_data/'
loader1 = TextLoader(path + 'samsung_outlook.txt', encoding='utf-8')
loader2 = TextLoader(path + 'nvidia_outlook.txt', encoding='utf-8')
# UnicodeDecodeError: 'cp949' codec can't decode byte 0xec in position 0: illegal multibyte sequence
# RuntimeError: Error loading ./_data/rag_data/samsung_outlook.txt
# 한글로 작성된 txt 파일은 컴퓨터가 읽을수있게 인코딩 방식을 지정해줘야 한다. (encoding='utf-8')

#02 문서를 자른다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 100,
    separators = ['\n\n', '\n', ' ', ''],     # 통상 이렇게 하는게 디폴트다.
)

split_doc1 = loader1.load_and_split(text_splitter)    # 청크 300, 오버랩 100     (파라미터 튜닝의 일환으로 언제든 내가 하고싶은데로 바꿀수있다.)
split_doc2 = loader2.load_and_split(text_splitter)

# 문서 개수 확인
# print(split_doc1)
# print(len(split_doc1), len(split_doc2))     # 9 9
'''

#03 임베딩.
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)
'''
######################## 여기부터 faiss ########################
faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query('hello world')))
# faiss_index = faiss.IndexFlatL2(1536)
print('FAISS 인덱스 초기화 준비완료')

# FAISS 벡터 저장소의 벡터 차원 수 (임베딩 차원 수)
print(faiss_index.d)     # 1536

faiss_db = FAISS(
    embedding_function=embeddings,
    index=faiss_index,
    docstore = InMemoryDocstore(),
    index_to_docstore_id={},
)
# 저장된 문서의 개수 확인.
print(faiss_db.index.ntotal)    # 0
######################## 여기까지 faissDB 준비완료 ########################
###########################################################################

db = FAISS.from_documents(
    documents=split_doc1 + split_doc2,
    embedding=embeddings,
)
'''
DB_PATH = './_db/FAISS17/'
# db.save_local(
#     folder_path=DB_PATH,
#     index_name='faiss_index17'       # 참고를 위해서 남겨둔다.
# )

db = FAISS.load_local(
    folder_path=DB_PATH,
    index_name='faiss_index17',
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)


print('=============================================')
# 문서 저장소 ID 확인
print(db.index_to_docstore_id)
print('=============================================')
# 저장된 결과 확인
print(db.docstore._dict)
print('=============================================')
# 유사도 검색
aaa = db.similarity_search('삼성전자 창업주에 대해 알려줘.', k=2)
print(aaa)