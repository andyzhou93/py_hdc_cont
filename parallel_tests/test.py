from ipyparallel import Client
client = Client()
dview = client[:]

a = 10
b = 5

dview.push({'a':a,'b':b})

def add():
    return a + b

print(dview.apply_sync(add))
