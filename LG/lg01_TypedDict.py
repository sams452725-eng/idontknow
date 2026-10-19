from typing import Dict, TypedDict

sample_dict : Dict[str, str] = {          # Dict 안에 keyvalu의 타입을 [str, str]을 쓰겠다. 강제성은 없고 힌트다.
    'name' : '김말똥',
    'age' : '39',
    'job' : '회사원'
}
print(sample_dict)
# {'name': '김말똥', 'age': '39', 'job': '회사원'}


##################################################################
class Person(TypedDict):
    name : str
    age : int
    job : str

person = Person(name='홍길동', age=40, job='도적')
print(person)
##################################################################

person2 : Person = {'name':'김개똥', 'age' : 20, 'job' : '전사'}
print(person2)
