from ipyparallel import Client
import numpy as np
import time

client = Client()
dview = client[:]

dview.push({'a':5,'b':10})

def add(i):
	import time
	import random
	random.seed(i)
	time.sleep(1)
	return (a + b)*random.randint(0,100), (a + b)

n = 10
start = time.time()
lview = client.load_balanced_view()
lview.block = True
print(lview.map(add, range(n)))
stop = time.time()
print('Elapsed: %f' % (stop - start))

start = time.time()
res = []
for i in range(n):
	res.append(add(i))
print(res)
stop = time.time()
print('Elapsed: %f' % (stop - start))