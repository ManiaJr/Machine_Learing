# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

epoch=0
flag=True
times=1


def successful(classA,classB):
    #global choice
    global w1,w2,fun1,fun2
    recall = np.zeros((2, 3))
    recall[0, :2] = 0.4 * np.random.rand(1, 2)
    recall[1, :2] = 0.4 * np.random.rand(1, 2) + 0.5
    
    if choice==4:
        recall[1, 2] = 0.4 * np.random.rand(1) + 0

    recall[:, 2] = -1
    plt.figure(2)
    plt.cla()
    plt.subplot(1,2,1)
    plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'kd')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('Axis x')
    plt.ylabel('Axis y')
    plt.title('Protypa')
    
    plt.subplot(1,2,2)
    plt.cla()
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f'Grafima Protipon-training-Epoch {epoch}')
    plt.plot(data[classA,0],data[classA,1],"mo",data[classB,0],data[classB,1],"kd")
    plt.plot(recall[0,0],recall[0,1],'mo')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.plot(recall[1,0],recall[1,1],'kd')
    
    x1 = np.arange(0, 1.1, 0.1)
    y11=-(w1[0,0]*x1 -w1[2,0])/w1[1,0]
    plt.plot(x1, y11, linewidth=3)
    
    x2 = np.arange(0, 1.1, 0.1)
    y12=-(w1[0,1]*x2 -w1[2,1])/w1[1,1]
    plt.plot(x2, y12, linewidth=3)
    
    plt.savefig('bp_20_pr_2a_b_03_f13_epoch_1000_recall.tif', format='tiff')
    
    u1 =np.zeros(n)
    y1 = np.zeros(3)
    u2 = np.zeros(n)
    y2 = np.zeros(n)
    for i in range(2):
        for j in range(2):
            u1[j]=np.dot(recall[i,:],w1[:,j])
            if fun1==1:
                y1[j]=1 / (1 + np.exp(-u1[j]))
            elif fun1==2:
                y1[j]=np.tanh(u1[j])
            elif fun1==3:
                y1[j]=u1[j]
        y1[2]=-1
        
        u2 = np.dot(y1,w2.T)
        if fun2==1:
            y2[i]=1 / (1 + np.exp(-u2))
        elif fun2==2:
            y2[i]=np.tanh(u2)
        elif fun2==3:
            y2[i]=u2
        
        if fun2==1:
            if y2[i]<=0.5:
                print('y2= ',y2[i],' recall ',i,' = ',recall[i,:],' in class 0')
            else:
                print('y2= ',y2[i],' recall ',i,' = ',recall[i,:],' in class 1')
        elif fun2==2:
            if y2[i]<=0:
                print('y2= ',y2[i],' recall ',i,' = ',recall[i,:],' in class 0')
            else:
                print('y2= ',y2[i],' recall ',i,' = ',recall[i,:],' in class 1')
        elif fun2==3:
            if y2[i]<=0:
                print('y2= ',y2[i],' recall ',i,' = ',recall[i,:],' in class 0')
            else:
                print('y2= ',y2[i],' recall ',i,' = ',recall[i,:],' in class 1')
    

def animate(i):
    global epoch,flag,times,fun1,fun2,mse
    while flag and epoch<=max_num_of_epochs:
        sfalma = 0;
        u1 =np.zeros(n)
        y1 = np.zeros(3)
        u2 = np.zeros(n)
        y2 = np.zeros(n)
        bhma1 = np.zeros(n)
        bhma2 = 0;
        for i in range(n):
            for j in range(2):
                u1[j] = np.dot(data[i,:],w1[:,j])
                if fun1==1:
                    y1[j]=1 / (1 + np.exp(-u1[j]))
                elif fun1==2:
                    y1[j]=np.tanh(u1[j])
                elif fun1==3:
                    y1[j]=u1[j]
            y1[2]=-1

            u2 = np.dot(y1,w2.T)
            if fun2==1:
                y2[i]=1/(1 + np.exp(-u2))
                bhma2=(1-y2[i]* y2[i]*(d[i]-y2[i]))
            elif fun2==2:
                y2[i]=np.tanh(u2)
                bhma2=(1-(y2[i])**2)*(d[i]-y2[i])
            elif fun2==3:
                y2[i]=u2
                bhma2=(d[i]-y2[i])
            
            if fun1==1:
                for j in range(2):
                    bhma1[j]=(1-y1[j])*y1[j]*(bhma2*w2[0,j])
            elif fun1==2:
                for j in range(2):
                    bhma1[j]=(1-(y1[j])**2)*(bhma2*w2[0,j])
            elif fun1==3:
                for j in range(2):
                    bhma1[j]=(bhma2*w2[0,j])
            
            for j in range(3):
                w2[0,j]=w2[0,j]+bhma*bhma2*y1[j]
            
            for j in range(2):
                for k in range(3):
                    w1[k,j]=w1[k,j]+bhma*bhma1[j]*data[i,k]
        
        for i in range(n):
            for j in range(2):
                u1[j]=np.dot(data[i,:],w1[:,j])
                if fun1==1:
                    y1[j]=1 / (1 + np.exp(-u1[j]))
                elif fun1==2:
                    y1[j]=np.tanh(u1[j])
                elif fun1==3:
                    y1[j]=u1[j]
            
            y1[2]=-1
            u2 = np.dot(y1,w2.T)
            if fun2==1:
                y2[i]=1 / (1 + np.exp(-u2))
            elif fun2==2:
                y2[i]=np.tanh(u2)
            elif fun2==3:
                y2[i]=u2
            
            sfalma=sfalma+(d[i]-y2[i])**2
        
        if sfalma/n<=min_sfalma:
            flag=False
        epoch=epoch+1
        print("sflama/n: ",sfalma/n," minsfalma: ",min_sfalma," epoch: ",epoch," flag = ",flag)
        mse.insert(epoch, sfalma/n)
        if fun2==1:
            classA=np.where(y2<=0.5)
            classB=np.where(y2>0.5)
        elif fun2==2:
            classA=np.where(y2<=0)
            classB=np.where(y2>0)
        elif fun2==3:
            classA=np.where(y2<=0)
            classB=np.where(y2>0)
        plt.figure(1)
        plt.subplot(2,2,2)
        plt.cla()
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f'Grafima Protipon-training-Epoch {epoch}')
        plt.plot(data[classA,0],data[classA,1],"mo",data[classB,0],data[classB,1],"kd")
        
        x1 = np.arange(0, 1.1, 0.1)
        y11=-(w1[0,0]*x1 -w1[2,0])/w1[1,0]
        plt.plot(x1, y11, linewidth=3)
        
        x2 = np.arange(0, 1.1, 0.1)
        y12=-(w1[0,1]*x2 - w1[2,1])/w1[1,1]
        plt.plot(x1, y12, linewidth=3)
        
        plt.pause(0.05)
        
        plt.subplot(2,2,3)
        plt.cla()
        #edw paizei na einai lathos
        y3 = np.arange(1, n+1)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.plot(np.arange(1,n//2+1),y3[0:n//2],"mo",np.arange(n//2+1,n+1),y3[n//2:n],"kd")
        plt.xlabel("Protypo")
        plt.ylabel("Exodos Y")
        plt.title("Protypa kai Exodoi")
        
        plt.pause(0.05)
        
        plt.subplot(2,2,4)
        plt.plot(mse)
        plt.xlabel("Epoch")
        plt.ylabel("mse")
        plt.title("MSE Graph")
        
        plt.show()
        
        plt.pause(0.05)
        print('epoch = ',epoch,' mse = ',mse)
        
        
    plt.savefig('bp_20_pr_2a_b_03_f13_epoch_last.tif', format='tiff')
    print("Press enter to recall: \n")
    if flag!=True and times==1:
        successful(classA,classB)
        times+=1
                 
#arxh
fig=plt.figure()
plt.subplots_adjust(wspace=0.5,hspace=0.5)
n=int(input('\n Plhthos apo dedomena(pollaplasio toy 8 ): '))
data=np.zeros((n,2))
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
col=[[-1]*1]*n
data=np.append(data,col,axis=1)

w1 = 2 * np.random.randn(3, 2) - 1
w2 = 2 * np.random.randn(1, 3) - 1
d=np.zeros(n)
mse=[]
fun1=0
while fun1<1 or fun1>3:
    print('\n')
    print('Transfer Function-Hidden Layer\n')
    print(' 1. Sigmoid\n 2. Tanh\n 3. Linear\n ')
    fun1=int(input('Give choice: '))
    
fun2=0
while fun2<1 or fun2>3:
    print('\n')
    print('Transfer Function-Hidden Layer\n')
    print(' 1. Sigmoid\n 2. Tanh\n 3. Linear\n ')
    fun2=int(input('Give choice: '))
    
if fun2==1:
    d[n//2:n]=1
else:
    d[0:n//2]=-1
    d[n//2:n]=1
    
bhma=float(input('\n bhma(0.1): '))
max_num_of_epochs=int(input('\n max apo epoxes: '))
min_sfalma=float(input("\n minimum sfalma:(0.1 peripou) "))

plt.subplot(2,2,1)
plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'kd')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Axis x')
plt.ylabel('Axis y')
plt.title('Protypa')

ani = FuncAnimation(fig, animate, frames=120, interval=100,repeat=False)