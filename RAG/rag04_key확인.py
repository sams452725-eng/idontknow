import os
key = os.getenv('OPENAI_API_KEY')      # 환경변수에 getenv를 통해서 'OPENAI_API_KEY'를 불러와라.

if key is None:
    print('OPENAI_API_KEY 없다!!')      # 없으면 해당 문구 반출
else:
    print('키 길이 :', len(key))        # 있으면 다음과 같이 반출  
    print('키 확인 :', key[:8] + '...' + key[-4:])


    