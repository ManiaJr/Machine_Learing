# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys


def gauss(x, c, sigma, i, j):
    return np.exp(-np.linalg.norm(x[i, 0:2] - c[j, 0:2])**2 / 2)


def cauchy(x, c, sigma, i, j):
    return 1 / (sigma * (np.linalg.norm(x[i, 0:2] - c[j, 0:2])**2 + sigma**2))


def polytetragoniki(x, c, sigma, i, j):
    return np.sqrt(np.linalg.norm(x[i, 0:2] - c[j, 0:2])**2 + sigma**2)

def successful():
    n=2
    data=np.ones((n,2))
    if choice == 1 or choice == 2 or choice == 3 or choice == 4:
        data[:n//2, 0:2] = 0.4 * np.random.rand(n//2, 2) + 0  
        
        if choice == 4:
            data[n//2:n, 0] = 0.4 * np.random.rand(n//2) + 0.5  
            data[n//2:n, 1] = 0.4 * np.random.rand(n//2) + 0
        else:
            data[n//2:n, 0:2] = 0.4 * np.random.rand(n//2, 2) + 0.5
    else:
        r = 0.08*np.random.randn(n//2)
        theta = 2*np.pi*np.random.rand(n//2)
        data[:n//2, 0] = 0.5 + r*np.cos(theta)
        data[:n//2, 1] = 0.5 + r*np.sin(theta)
        
        r = 0.6+0.1*np.random.randn(n//2)
        theta = 2*np.pi*np.random.rand(n//2)
        data[n//2:, 0] = 0.5 + r*np.cos(theta)
        data[n//2:, 1] = 0.5 + r*np.sin(theta)
    
    x = np.zeros((n, neyrones+1))
    for i in range(n):
        for j in range(neyrones):
            if epilogh == 1:
                x[i, j] = gauss(data, c, sigma, i, j)
            elif epilogh == 2:
                x[i, j] = cauchy(data, c, sigma, i, j)
            elif epilogh == 3:
                x[i, j] = polytetragoniki(data, c, sigma, i, j)

    x[:, neyrones] = 1
    
    for i in range(n):
        u = np.dot(x[i,:],w)
        y[i] = u
        delta[i]=d[i]-y[i]
        if y[i] <= 0:
            print(f'Pattern #{i+1} = {data[:,i]} in class -1')
        else:
            print(f'Pattern #{i+1} = {data[:,i]} in class 1')


def animate(i):
    global epoch,y,mse,times,x,mserror,flag
    while flag and epoch<=max_num_of_epochs:
        for i in range(n):
            u = np.dot(x[i,:],w)
            y[i] = u
            delta[i]=d[i]-y[i]
            for j in range(neyrones+1):
                w[j]=w[j]+bhma*delta[i]*x[i,j]
        sfalma=0
        for i in range(n):
            u = np.dot(x[i,:],w)
            y[i] = u
            delta[i]=d[i]-y[i]
            sfalma=sfalma+delta[i]**2
        if sfalma/n<=min_sfalma:
            flag=False   
        epoch=epoch+1
        mse.insert(epoch, sfalma/n)
        classA=np.where(y<=0)
        classB=np.where(y>0)
        plt.figure(2)
        plt.subplots_adjust(wspace=0.5,hspace=0.5)
        plt.subplot(2,2,1)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'kd')
        plt.xlabel('Axis x')
        plt.ylabel('Axis y')
        plt.title('Protypa')
        plt.pause(0.05)
        
        plt.subplot(2,2,2)
        plt.cla()
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot(data[classA,0],data[classA,1],"mo",data[classB,0],data[classB,1],"kd")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f'Protypa - Train - Epoch = {epoch}')
        
        
        plt.subplot(2,2,3)
        plt.cla()
        plt.plot(np.arange(1,n//2+1),y[0:n//2],"mo",np.arange(n//2+1,n+1),y[n//2:n],"kd")
        plt.xlabel("Protypo")
        plt.ylabel("Exodos Y")
        plt.title("Protypa kai Exodoi")
        plt.pause(0.05)
        
        plt.subplot(2,2,4)
        plt.plot(mse,'c')
        plt.xlabel("Epoch")
        plt.ylabel("mse")
        plt.title(f'mse = {sfalma/n}')
        plt.pause(0.05)
        
    if flag!=True and times==1:
        successful()
        times+=1


#arxh
fig=plt.figure()
plt.subplots_adjust(wspace=0.5,hspace=0.5)
n=int(input('\n Plhthos apo dedomena(pollaplasio toy 4 ): '))
neyrones=int(input('\n Dwse arithmo antagwnistikwn neyrwnwn: '))
        
data=np.ones((n,2))
choice=1
while choice!=0:
    print("Epeleje ena apo ta parakatw: ")
    print("\n1.Grammika Diaxwrisima Protupa\n2.Mh Grammika Diaxwrisima Protupa (Klash 0 sth Gwnia)\n3.Mh Grammika Diaxwrisima Protupa (Klash 0 sto Kentro)\n4.Mh Grammika Diaxwrisima Protupa (XOR)\n5.Mh Grammika Diaxwrisima Protupa (Klash 0 mesa sthn Klash 1)\n0.Exit\n")
    choice=int(input("Epilogh: "))
    while choice<0 or choice>5:
        choice=input("Epilogh")
    if choice==1:
        data[:n//2, :2] = 0.4 * np.random.rand(n//2, 2)
        data[n//2:n, :2] = 0.5 * np.random.rand(n//2, 2)+0.5
        break
    elif choice==2:
        data[:n//2, :2] = 0.5 * np.random.rand(n//2, 2)
        data[n//2:n, :2] = 0.6 * np.random.rand(n//2, 2)+0.4
        break
    elif choice==3:
        data[:n//2, :2] = 0.4 * np.random.rand(n//2, 2)
        data[n//2:3*n//4, 0:1] = 0.9 * np.random.rand(n//4, 1)
        data[n//2:3*n//4, 1:2] = 0.4 * np.random.rand(n//4, 1)+ 0.5
        data[3*n//4:n, 0:1] = 0.4 * np.random.rand(n//4, 1)+0.5
        data[3*n//4:n, 1:2] = 0.4 * np.random.rand(n//4, 1)
        break
    elif choice==4:
        data[:n//4, :2] = 0.4 * np.random.rand(n//4, 2)
        data[n//4:n//2, :2] = 0.4 * np.random.rand(n//4, 2) + 0.5
        data[n//2:3*n//4, 0:1] = 0.4 * np.random.rand(n//4, 1) + 0.5
        data[n//2:3*n//4, 1:2] = 0.4 * np.random.rand(n//4, 1) 
        data[3*n//4:n, 0:1] = 0.4 * np.random.rand(n//4, 1) 
        data[3*n//4:n, 1:2] = 0.4 * np.random.rand(n//4, 1) + 0.5
        break
    elif choice==5:
        r = 0.08*np.random.randn(n//2)
        theta = 2*np.pi*np.random.rand(n//2)
        data[:n//2, 0] = 0.5 + r*np.cos(theta)
        data[:n//2, 1] = 0.5 + r*np.sin(theta)
        
        r = 0.6+0.1*np.random.randn(n//2)
        theta = 2*np.pi*np.random.rand(n//2)
        data[n//2:, 0] = 0.5 + r*np.cos(theta)
        data[n//2:, 1] = 0.5 + r*np.sin(theta)
        break
    elif choice==0:
        break
    
d = np.ones((n, 1))
d[:n//2] = -1
w = np.random.rand(neyrones + 1, 1)

plt.subplot(1,2,1)
plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'bd')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Axis x')
plt.ylabel('Axis y')
plt.title('patern')

c = 0.9 * np.random.rand(neyrones, 2) + 0
c_old = np.zeros((neyrones, 2))

while not np.array_equal(c_old, c):
    c_old = c
    deiktes = np.zeros((n, 1))
    
    for i in range(n):
        apostash = np.zeros((1, neyrones))
        
        for j in range(neyrones):
            apostash[0, j] = np.linalg.norm(data[i, 0:2] - c[j, 0:2]) ** 2
        
        deiktes[i, 0] = np.argmin(apostash)
    
    c = np.zeros((neyrones, 2))
    count = np.zeros((neyrones, 2))
    
    for i in range(n):
        c[int(deiktes[i, 0]), 0:2] += data[i, 0:2]
        count[int(deiktes[i, 0])] += 1
    
    for j in range(neyrones):
        if (count[j] != 0).any():
            c[j, :] /= count[j]

plt.subplot(1,2,2)
plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'bd')
plt.plot(c[:, 0], c[:, 1], 'ko', markersize=12, markerfacecolor='k')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Axis x')
plt.ylabel('Axis y')
plt.title('patern')


print('\n')
print('ÊÝíôñá ok - Hit Enter to Continue : ', end='')
sys.stdin.readline()


apostash1 = np.zeros(neyrones * neyrones)
for i in range(neyrones):
    for j in range(neyrones):
        apostash1[(i-1)*neyrones+j] = np.linalg.norm(c[i, 0:2] - c[j, 0:2]) ** 2

megisto = np.max(apostash1)
thesi_max = np.argmax(apostash1)
sigma = megisto / np.sqrt(2 * n)

print('\n')
epilogh=0
while epilogh<1 or epilogh>3:
    print('Epeleje RBF(1.Gauss 2.Cauchy 3.Polytetragoniki): ')
    epilogh=int(input("Epilogh: "))
    
bhma=float(input('\n bhma(0.05): '))
max_num_of_epochs=int(input('\n max apo epoxes: '))
min_sfalma=float(input("\n minimum sfalma:(0.15 peripou) "))

mse=[]
epoch=0
flag=True
times=1
x = np.zeros((n, neyrones+1))
y=np.zeros(n)
delta = np.zeros(n+1)

for i in range(n):
    for j in range(neyrones):
        if epilogh == 1:
            x[i, j] = gauss(data, c, sigma, i, j)
        elif epilogh == 2:
            x[i, j] = cauchy(data, c, sigma, i, j)
        elif epilogh == 3:
            x[i, j] = polytetragoniki(data, c, sigma, i, j)

x[:, neyrones] = 1


ani = FuncAnimation(fig, animate, frames=120, interval=100, repeat=False)
plt.show()


