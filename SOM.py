# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
epoch=0

def animate(i):
    global epoch,syek,geit
    while epoch<=200 and syek>0.001:
        w_old=w
        winners=np.zeros(n)
        for i in range(n):
            apostash=np.zeros(neyrones)
            for j in range(neyrones):
                apostash[j] = np.linalg.norm(data[i, 0:2] - w[j, 0:2])
            thesi = np.argmin(apostash)
            elax = np.min(apostash)
            winners[i]=thesi
            w[thesi,:]=w[thesi,:]+syek*(data[i,:2]-w[thesi,:2])
            plt.figure(1)
            plt.cla()
            plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'bd')
            plt.plot(w[:neyrones, 0], w[:neyrones, 1], 'ko', markersize=12, markerfacecolor='k')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.xlabel('Axis x')
            plt.ylabel('Axis y')
            plt.title(f'Protypa-Epoch {epoch}-Synapsewn {syek}')
            plt.pause(0.005)
        syek=syek*(1-epoch/200)
        epoch=epoch+1
    print("Prwth fash Ekpaideyshs\n")
    for i in range(1,201):
        w_old=w
        winners=np.zeros(n)
        for j in range(n):
            apostash=np.zeros(neyrones*neyrones)
            for k in range(neyrones*neyrones):
                apostash[k] = np.linalg.norm(data[j, 0:2] - w[k, 0:2])
            thesi = np.argmin(apostash)
            elax = apostash[thesi]
            winners[j]=thesi
            k=thesi%neyrones
            if k==0:
                k=neyrones
            flag=(thesi-k)/neyrones+1
            geitones=np.zeros(neyrones*neyrones)
            l=1
            temp = np.zeros((neyrones*neyrones, 2))
            for jei in range(int(flag)-geit,int(flag)+geit):
                for kei in range(k-geit,k+geit):
                    if jei>0 and kei>0 and jei<=neyrones and kei<=neyrones:
                        jeikei=(jei-1)*neyrones+kei
                        if jeikei>0 and jeikei<=neyrones*neyrones:
                            w[jeikei,:]=w[jeikei,:]+syek*(data[j,:2]-w[jeikei,:2])
                            temp[l,0]=jei
                            temp[l,1]=kei
                            geitones[l]=jeikei
                            l=l+1
            if (i==1) or (i==10) or (i==50) or (i==100) or (i==200):
                plt.figure(1)
                plt.subplot(2,2,1)
                plt.cla()
                plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'bd')
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.xlabel('Axis x')
                plt.ylabel('Axis y')
                plt.title(f'Protypa-Epoch {i}')
                plt.pause(0.005)
                
                plt.subplot(2,2,2)
                plt.plot(w[:neyrones*neyrones, 0], w[:neyrones*neyrones, 1], 'ko', markersize=12, markerfacecolor='k')
                plt.plot(w[np.where(geitones[:l-1]), 0], w[np.where(geitones[:l-1]), 1], 'go', markersize=12)
                plt.plot(w[thesi,0],w[thesi,1],'ro',markersize=14)
                plt.xlabel('Axis x')
                plt.ylabel('Axis y')
                plt.title(f'graghma nikhth {winners[j]} - Geitones')
                plt.pause(0.005)
                
                plt.subplot(2,2,3)
                jj=0
                xy=np.zeros((neyrones**2,2))
                for o in range(neyrones):
                    for k in range(neyrones):
                        xy[jj,0]=o
                        xy[jj,1]=k
                        jj=jj+1
                plt.plot(xy[:,0],xy[:,1],'ks',markersize=8)
                plt.plot(temp[0:l-1,0],temp[0:l-1,1],'go',markersize=12)
                k=thesi%neyrones
                if k==0:
                    k=neyrones
                flag=(thesi-k)/neyrones+1
                plt.plot(flag,k, markersize=14)
                plt.xlabel('Axis x')
                plt.ylabel('Axis y')
                plt.title(f'Topologiko grafhma nikhth {winners[j]} - Geitones')
                plt.pause(0.005)
                
                plt.subplot(2,2,4)
                plt.plot(range(0,j//2),winners[0:j//2],'mo',range(j//2,j),winners[j//2:j],'bd')
                plt.xlabel('Arithmos apo protypa')
                plt.ylabel('Nikhths')
                plt.title('Grafhma nikhth ana protypo')
                plt.pause(0.005)
        if (i % 1 == 0) or (i % 1 == 0) or (i % 1 == 0) or (i % 1 == 0) or (i % 1 == 0):
            plt.figure(2)
            plt.subplots_adjust(wspace=0.5,hspace=0.5)
            plt.subplot(2,2,1)
            plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'bd')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.xlabel('Axis x')
            plt.ylabel('Axis y')
            plt.title(f'Protypa-Epoch {i}')
            plt.pause(0.005)
            
            plt.subplot(2,2,2)
            plt.plot(w[:neyrones*neyrones, 0], w[:neyrones*neyrones, 1], 'ko', markersize=12, markerfacecolor='k')
            plt.plot(w[np.where(geitones[:l-1]), 0], w[np.where(geitones[:l-1]), 1], 'go', markersize=12)
            plt.plot(w[thesi,0],w[thesi,1],'ro',markersize=14)
            plt.xlabel('Axis x')
            plt.ylabel('Axis y')
            plt.title(f'graghma nikhth {winners[j]} - Geitones')
            plt.pause(0.005)
            
            plt.subplot(2,2,3)
            jj=0
            xy=np.zeros((neyrones**2,2))
            for o in range(neyrones):
                for k in range(neyrones):
                    xy[jj,0]=o
                    xy[jj,1]=k
                    jj=jj+1
            plt.plot(xy[:,0],xy[:,1],'ks',markersize=8)
            
            for counter in range(1,j):
                k=winners[counter]%neyrones
                if counter==0:
                    counter=neyrones
                flag=(winners[counter]-k)/neyrones+1
                if counter<=j//2:
                    plt.plot(flag,k,'bo',markersize=14)
                else:
                    plt.plot(flag,k,'mo',markersize=14)
            
            plt.xlabel('Arithmos apo protypa')
            plt.ylabel('Nikhths')
            plt.title('Grafhma nikhth ana protypo')
            plt.pause(0.005)
            
            plt.subplot(2,2,4)
            plt.plot(range(0,j//2),winners[0:j//2],'mo',range(j//2,j),winners[j//2:j],'bd')
            plt.xlabel('Arithmos apo protypa')
            plt.ylabel('Nikhths')
            plt.title('Grafhma nikhth ana protypo')
            plt.pause(0.005)
        if i % round(200 / (geit - 1)) == 0:
            geit=geit-1
    print("Deyterh fash Ekpaideyshs\n")
    epoch=0
    while epoch<=500*neyrones*neyrones and syek>=0.1 and geit>=0:
        w_old=w
        winners=np.zeros(n)
        for j in range(n):
            apostash=np.zeros(neyrones*neyrones)
            for k in range(neyrones*neyrones):
                apostash[k] = np.linalg.norm(data[j, :2] - w[k, :2])
            thesi = np.argmin(apostash)
            elax = np.min(apostash)
            winners[j]=thesi
            k=thesi%neyrones
            if k==0:
                k=neyrones
            flag=(thesi-k)/neyrones+1
            geitones=np.zeros(neyrones*neyrones)
            l=1
            temp=np.zeros((neyrones,2))
            for jei in range(int(flag)-geit,int(flag)+geit):
                for kei in range(k-geit,k+geit):
                    if jei>0 and kei>0 and jei<=neyrones and kei<=neyrones:
                        jeikei=(jei-1)*neyrones+kei
                        if jeikei>0 and jeikei<=neyrones*neyrones:
                            w[jeikei,:]=w[jeikei,:]+syek*(data[j,:2]-w[jeikei,:2])
                            temp[l,0]=jei
                            temp[l,1]=kei
                            geitones[l]=jeikei
                            l=l+1
            if epoch%50*neyrones*neyrones==0:
                print('Epoxh=',epoch,' bhma=',syek,' Geitonia=',geit,' protypo=',j,' nikhths=',w[thesi,0],w[thesi,1])
                plt.figure(1)
                plt.cla()
                plt.subplot(2,2,1)
                plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'bd')
                plt.xlim([0,1])
                plt.ylim([0,1])
                plt.xlabel('Axis x')
                plt.ylabel('Axis y')
                plt.title(f'Protypa-Epoch {epoch}')
                plt.pause(0.005)
                
                plt.subplot(2,2,2)
                plt.plot(w[:neyrones*neyrones, 0], w[:neyrones*neyrones, 1], 'ks', markersize=12, markerfacecolor='k')
                plt.plot(w[geitones[:l-1], 0], w[geitones[:l-1], 1], 'go', markersize=12)
                plt.plot(w[thesi,0],w[thesi,1],'ro',markersize=14)
                plt.xlabel('Axis x')
                plt.ylabel('Axis y')
                plt.title(f'graghma nikhth {winners[j]} - Geitones')
                plt.pause(0.005)
                
                plt.subplot(2,2,3)
                jj=0
                xy=np.zeros((neyrones**2,2))
                for o in range(neyrones):
                    for k in range(neyrones):
                        xy[jj,0]=o
                        xy[jj,1]=k
                        jj=jj+1
                plt.plot(xy[:,0],xy[:,1],'ks',markersize=8)
                plt.plot(temp[0:l-1,0],temp[0:l-1,1],'go',markersize=12)
                k=thesi%neyrones
                if k==0:
                    k=neyrones
                flag=(thesi-k)/neyrones+1
                plt.plot(flag,k, markersize=14)
                plt.xlabel('Axis x')
                plt.ylabel('Axis y')
                plt.title(f'Topologiko grafhma nikhth {winners[j]} - Geitones')
                plt.pause(0.005)
                
                plt.subplot(2,2,4)
                plt.plot(range(0,j//2),winners[0:j//2],'mo',range(j//2,j),winners[j//2:j],'bd')
                plt.xlabel('Arithmos apo protypa')
                plt.ylabel('Nikhths')
                plt.title('Grafhma nikhth ana protypo')
                plt.pause(0.005)
        if epoch%50*neyrones*neyrones==0:
            print('Epoxh=',epoch,' bhma=',syek,' Geitonia=',geit,' protypo=',j,' nikhths=',w[thesi,0],w[thesi,1])  
            plt.figure(1)
            plt.cla()
            plt.subplot(2,2,1)
            plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'bd')
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.xlabel('Axis x')
            plt.ylabel('Axis y')
            plt.title(f'Protypa-Epoch {epoch}')
            plt.pause(0.005)
            
            plt.subplot(2,2,2)
            plt.plot(w[:neyrones*neyrones, 0], w[:neyrones*neyrones, 1], 'ks', markersize=12, markerfacecolor='k')
            plt.plot(w[geitones[:l-1], 0], w[geitones[:l-1], 1], 'go', markersize=12)
            plt.plot(w[thesi,0],w[thesi,1],'ro',markersize=14)
            plt.xlabel('Axis x')
            plt.ylabel('Axis y')
            plt.title(f'graghma nikhth {winners[j]} - Geitones')
            plt.pause(0.005)
            
            plt.subplot(2,2,3)
            jj=0
            xy=np.zeros((neyrones**2,2))
            for o in range(neyrones):
                for k in range(neyrones):
                    xy[jj,0]=o
                    xy[jj,1]=k
                    jj=jj+1
            plt.plot(xy[:,0],xy[:,1],'ks',markersize=8)
            
            for counter in range(1,j):
                k=winners[counter]%neyrones
                if counter==0:
                    counter=neyrones
                flag=(winners[counter]-k)/neyrones+1
                if counter<=j//2:
                    plt.plot(flag,k,'bo',markersize=14)
                else:
                    plt.plot(flag,k,'mo',markersize=14)
            
            plt.xlabel('Arithmos apo protypa')
            plt.ylabel('Nikhths')
            plt.title('Grafhma nikhth ana protypo')
            plt.pause(0.005)
            
            plt.subplot(2,2,4)
            plt.plot(range(0,j//2),winners[0:j//2],'mo',range(j//2,j),winners[j//2:j],'bd')
            plt.xlabel('Arithmos apo protypa')
            plt.ylabel('Nikhths')
            plt.title('Grafhma nikhth ana protypo')
            plt.pause(0.005)
        if epoch+800%50*neyrones*neyrones==0:
            syek=syek-0.1
    print('\nTelikes Synapseis:\n')
    print(w)
    print('\nTelikoi Nikhtes Ekpaideyshs:\n')
    print(winners)
        
#arxh
fig=plt.figure()
plt.subplots_adjust(wspace=0.5,hspace=0.5)
n=int(input('\n Plhthos apo dedomena(pollaplasio toy 4 ): '))
neyrones=int(input('\n Dwse arithmo antagwnistikwn neyrwnwn: '))
syek=float(input('\n bhma(0.1): '))
geit=1
w = np.random.rand(neyrones*neyrones, 2)
w_old=np.random.rand(neyrones, 2)
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


y=np.zeros(n)

plt.plot(data[0:n//2, 0], data[0:n//2, 1], 'mo', data[n//2:, 0], data[n//2:, 1], 'bd')
plt.plot(w[:neyrones, 0], w[:neyrones, 1], 'ko', markersize=12, markerfacecolor='k')
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('Axis x')
plt.ylabel('Axis y')
plt.title(f'Protypa kai Epoch {epoch} kai Synapsewn {syek}')
epoch=epoch+1

ani = FuncAnimation(fig, animate, frames=120, interval=100,repeat=False)