from langchain_openai import ChatOpenAI

openai_api_key = 'sk-proj-YOUR_API_KEY_HERE'
# 실무에선 절대 key를 노출하면 안된다.(github 같은 곳에 올라가면 큰일난다.)

llm = ChatOpenAI(
    model_name='gpt-5.6-terra',
    temperature=0,
    openai_api_key = openai_api_key,
)

# 현재 상태에서는 메모리 기능이 없기 때문에 각 질문이 저장이 안되서 매번 답변이 다르다.
# 그리고 메모리는 직전 대화를 메모리하기도 하지만 직전까지 한 모든 대화를 누적해서 프롬프트화 되서 다음 질문에 들어가기 때문에 컨텍스트가 길어져서 토큰을 더 먹는다.

response = llm.invoke('안녕하세요.')      # Chatgpt에서 '안녕하세요.'를 입력하는 것과 같은 행위이다.
# print(response)
# content='안녕하세요! 무엇을 도와드릴까요?'additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 9, 'total_tokens': 23, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0, 'text_tokens': None}, 'prompt_tokens_details': {'audio_tokens': 0, 'cache_write_tokens': 0, 'cached_tokens': 0, 'image_tokens': None, 'text_tokens': None}}, 'model_provider': 'openai', 'model_name': 'gpt-5.6-terra', 'system_fingerprint': None, 'id': 'chatcmpl-ETdLjDN5DlGns5AXWwSjq7eKbbE7s', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None} id='lc_run--01a0efe9-e3a7-71e3-aa11-f7983e297bb1-0' tool_calls=[] invalid_tool_calls=[] usage_metadata={'input_tokens': 9, 'output_tokens': 14, 'total_tokens': 23, 'input_token_details': {'audio': 0, 'cache_read': 0, 'cache_creation': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}
print(response.content)    # 답변 중에 'content' 부분만을 출력해서 보여준다.
# 안녕하세요! 무엇을 도와드릴까요?
