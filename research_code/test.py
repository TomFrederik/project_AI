
def func():
    a = [1,2,3]
    for i in a:
        yield i

iterator = func()

print(next(iterator))
print(next(iterator))
print(next(iterator))
print(next(iterator))
