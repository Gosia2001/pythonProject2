import random
import time
import numpy as np
import matplotlib.pyplot as plt

print('zad1a')
def potęga_cyfr(n):
    lista1=[]
    for x in range(n, n+1):
        lista1.append(n**2)
        t = time.time()
        print(lista1)
        print(time.time() - t)
potęga_cyfr(1000000)
t = time.time()
q = time.time() - t
print('zad1b')
def potega(n):
  lista2 = [a**2 for a in range(n)]
  t = time.time()
  e = time.time() - t
  print(lista2)
  print(time.time() - t)
potega(1000000)
t = time.time()
e = time.time() - t
print('zad1c')
def potega(n):
    y = np.ndarray(shape=(n), dtype=float)
    for i in range(n):
        y[i]= i**2
        t = time.time()
        w = time.time() - t
    print(y)
    print(time.time() - t)
potega(1000000)
t = time.time()
w = time.time() - t
print(q, w, e)
print('zad2')
print(np.ones((4,4), dtype=bool))
print('zad3')
a = np.arange(10)
b = np.ones([10])
y = (a.reshape((2,5)))
x = (b.reshape((2,5)))
print(x)
print(y)
print(np.concatenate((x,y), axis=1))
print('zad4')
import random
list[0, 1]
x = np.array(list)
range(0, 1)
for x in range(0,1):
    prob = np.random.random(20)
    print(prob)
    if any(prob)> 0.5 and any(prob) < 0.9:
        print(prob)
print('zad5')
p = np.loadtxt('dih_sample.dat')
f = (p-min(p))/(max(p)-min(p))
print(f)
np.savetxt('normalizacja.dat',f,fmt="%.4f")
print('zad6')
wsp = int(input("Podaj wspolczynnik a: "))
x1 = np.arange(-10,1,1)
x2 = np.arange(0,11,1)
y1 = np.array((-(wsp*x1))/3)
y2 = np.array(x2*x2/3)
plt.plot(x2, y2, label="y = (x^2)/3")
plt.plot(x1, y1, label="y = -(a*x)/3")
plt.xticks(np.arange(-10,11,2))
plt.rc('font',size=8)
plt.grid(True)
plt.legend()
plt.xlim(-10,10)
plt.ylabel("y")
plt.xlabel("x")
plt.title("Wykres funkcji to: y = f(x)")
plt.show()