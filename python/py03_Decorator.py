
# @(Deco)는 한마디로 함수를 불러오는 함수
# 즉, return 값을 함수로 받아오는 기능이다.


def hello():
    print('안녕하세요.')

hello()

print('='*50,'1', '='*50)
def deco(func):
    def wrapper():
        print('함수 실행전')
        func()
        print('함수 실행후')
    return wrapper

@deco
def hello2():
    print('안녕하세요2.')

hello2()