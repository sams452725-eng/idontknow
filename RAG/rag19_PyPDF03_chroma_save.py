import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter, CharacterTextSplitter
from langchain_chroma import Chroma
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

from dotenv import load_dotenv
load_dotenv()
api_key = os.environ['MONOROUTER_API_KEY'].strip()
base_url = 'https://monogpt.kr/api/monorouter/v1'


#01 데이터를 불러온다.
path = './_data/'
pdf_loader = PyPDFLoader(path + 'attention is all you needs.pdf')
# pdf_docs = pdf_loader.load()

# print(type(pdf_docs))    #<class 'list'>
# print(len(pdf_docs))     # 15(pdf 페이지 수)
# print(pdf_docs)

# print('=============================================')
# print(pdf_docs[0])
# print('=============================================')


#02 문서를 자른다.

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    separators = ['\n\n', '\n', ' ', ''],
)

split_pdf = pdf_loader.load_and_split(text_splitter)



#03 임베딩.
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small',
    api_key=api_key,
    base_url=base_url,
)
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings( 
#     model_name='BAAI/bge-m3',               # 임베딩 백터의 차원 : 1024
#     model_kwargs={
#         'device' : 'cpu',
#         'local_files_only' : True,
#     }
# )
# from langchain_huggingface.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings( 
#     model_name='Qwen/Qwen3-Embedding-0.6B',               # 임베딩 백터의 차원 : 1024
#     model_kwargs={
#         'device' : 'cpu',
#         'local_files_only' : True,
#     }
# )
'''
######################## 여기부터 faiss ########################
faiss_index = faiss.IndexFlatL2(len(embeddings.embed_query('hello world')))
# faiss_index = faiss.IndexFlatL2(1536)
print('FAISS 인덱스 초기화 준비완료')

# FAISS 벡터 저장소의 벡터 차원 수 (임베딩 차원 수)
print(faiss_index.d)     # 1536

db = FAISS.from_documents(
    documents=split_pdf,
    embedding=embeddings,
)
DB_PATH = './_db/FAISS19/'
db.save_local(
    folder_path=DB_PATH,
    index_name='faiss_index19'
)
'''


DB_PATH = './_db/Chroma19/'

#저장
db = Chroma.from_documents(
    documents=split_pdf,
    embedding=embeddings,
    persist_directory=DB_PATH,
    collection_name='chroma19',
)