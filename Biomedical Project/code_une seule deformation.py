
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

N,M, L,l =40,40, 5.32*1e-2,7.54*1e-3
x = L/N
y = l/M
C =  1.81*1e-5
P1=15*1e-2
P2=0
xmin, xmax, ymax = 20, 26, 8

def be(i,j,N):
    return (j)*N + i
def re(k,N):
    return (k%N,k//N)


def deformer(Xmin,Xmax,Ymax):
    L=[]
    x=Xmin
    y=0
    while Xmin <= x <= Xmax and y <= Ymax :

        L.append((x,y))
        if x==Xmax :
            x=Xmin
            y+=1
            continue
        x+=1

    return L

Q = deformer(xmin, xmax, ymax)
def matrixes(N,M,C,x,y,P1,P2,Q, xmin, xmax, ymax):
    Y=y**2
    X=x**2
    B= np.zeros((3*N*M,1))
    mat = np.zeros((3*N*M,3*N*M))
    for j in range(N*M):
        a=re(j,N)[0]
        b=re(j,N)[1]
        if a!=0 and a!=N-1 and b!=0 and b!=M-1 and re(j,N) not in Q  :
            if  a==N-2:
            # 1 er eq
                mat[j][be(a+1,b,N)] = 1/X
                mat[j][be(a-1,b,N)] = 1/X
                mat[j][be(a,b+1,N)] = 1/Y
                mat[j][be(a,b-1,N)] = 1/Y
                mat[j][j] = -2*((1/Y)+(1/X))
                mat[j][be(a,b,N)+2*N*M] = -1/(C*x)
                mat[j][be(a-1,b,N)+2*N*M] = 1/(C*x)
            # 2 er eq
                mat[j+M*N][be(a+1,b,N)+M*N] = 1/X
                mat[j+M*N][be(a-1,b,N)+M*N] = 1/X
                mat[j+M*N][be(a,b+1,N)+M*N] = 1/Y
                mat[j+M*N][be(a,b-1,N)+M*N] = 1/Y
                mat[j+M*N][j+M*N] = -2*((1/Y)+(1/X))
                mat[j+M*N][be(a,b+1,N)+2*M*N] =- 1/(2*C*y)
                mat[j+M*N][be(a,b-1,N)+2*M*N] = 1/(2*C*y)
                #,dea
                mat[j+2*M*N][be(1,b,N)+2*M*N] = -1
                mat[j+2*M*N][be(0,b,N)+2*M*N] = 1
                mat[j+2*M*N][be(a+1,b,N)+2*M*N] = 1
                mat[j+2*M*N][be(a,b,N)+2*M*N] = -1


                mat[j+2*M*N][be(a,b,N)+M*N] = 1/(y)
                mat[j+2*M*N][be(a,b-1,N)+M*N] = -1/(y)
                mat[j+2*M*N][be(a,b,N)] = 1/(x)
                mat[j+2*M*N][be(a-1,b,N)] = -1/(x)
            else :
            # 1 er eq
                mat[j][be(a+1,b,N)] = 1/X
                mat[j][be(a-1,b,N)] = 1/X
                mat[j][be(a,b+1,N)] = 1/Y
                mat[j][be(a,b-1,N)] = 1/Y
                mat[j][be(a,b,N)] = -2*((1/Y)+(1/X))
                mat[j][be(a,b,N)+2*N*M] = -1/(C*x)
                mat[j][be(a-1,b,N)+2*N*M] = 1/(C*x)
            # 2 er eq
                mat[j+M*N][be(a+1,b,N)+M*N] = 1/X
                mat[j+M*N][be(a-1,b,N)+M*N] = 1/X
                mat[j+M*N][be(a,b+1,N)+M*N] = 1/Y
                mat[j+M*N][be(a,b-1,N)+M*N] = 1/Y
                mat[j+M*N][j+M*N] = -2*((1/Y)+(1/X))
                mat[j+M*N][be(a,b+1,N)+2*M*N] =- 1/(2*C*y)
                mat[j+M*N][be(a,b-1,N)+2*M*N] = 1/(2*C*y)
            # 3 er eq
                mat[j+2*M*N][be(a,b+1,N)+M*N] = 1/(y)
                mat[j+2*M*N][be(a,b,N)+M*N] = -1/(y)
                mat[j+2*M*N][be(a+1,b,N)] = 1/(x)
                mat[j+2*M*N][be(a,b,N)] = -1/(x)

        if a==0:
            #eq _P1
            mat[j+2*M*N][j+2*M*N] = 1
            B[j+2*M*N]=P1
            if b!=0 and b!=M-1:
              # periodicitée de vitesse
              # ux
                mat[j][j] = 1
                mat[j][j+N-1] = -1
               #uy
                mat[j+M*N][be(N-1,b,N)+ M*N] = -1
                mat[j+M*N][be(a,b,N)+ M*N] = 1

        if a==N-1:
            #eq _P2
            mat[j+2*M*N][be(a,b,N)+2*M*N] = 1
            B[j+2*M*N]=P2
            if b!=0 and b!=M-1:
               #eq periodicitée de derivée de vitesse
               #ux
                mat[j][be(N-1,b,N)] = 1
                mat[j][be(N-2,b,N)] = -1
                mat[j][be(1,b,N)] =-1
                mat[j][be(0,b,N)] = 1
              #uy
                mat[j+M*N][be(N-1,b,N)+M*N] = 1
                mat[j+M*N][be(N-2,b,N)+M*N] =-1
                mat[j+M*N][be(1,b,N)+M*N] =-1
                mat[j+M*N][be(0,b,N)+M*N] = 1

        if b ==0 :
            #ux =0 en bas
            mat[j][j] = 1
            #uy =0 en bas
            mat[j+M*N][j+M*N] = 1
            if  a!=0 and a!= N-1 :
                if re(j,N) not in Q:
                    # 8 er eq derivée normale
                    mat[j+2*M*N][be(a,b,N)+M*N] = -1
                    mat[j+2*M*N][be(a,b+1,N)+M*N] = 1




        if b== M-1:
            #ux =0 en haut
            mat[j][j] = 1
             #uy =0 en haut
            mat[j+M*N][j+M*N] = 1
            if  a!=0 and a!= N-1  :
                mat[j+2*M*N][be(a,b-1,N)+M*N] = -1
                mat[j+2*M*N][be(a,b,N)+M*N] = 1
                #tar
                mat[j+2*M*N][be(a,b-2,N)+2*M*N] = 1
                mat[j+2*M*N][be(a,b,N)+2*M*N] = 1
                mat[j+2*M*N][be(a,b-1,N)+2*M*N] = -2

        #Déformation

        #U=0
        if re(j,N) in Q:
            mat[j][be(a,b,N)] = 1
            mat[j+M*N][be(a,b,N)+M*N] = 1
         # div
        if b == ymax and xmin==a:
            mat[j+2*M*N][be(a,b,N)] =1
            mat[j+2*M*N][be(a-1,b,N)] = -1
            mat[j+2*M*N][be(a,b+1,N)+M*N] = 1
            mat[j+2*M*N][be(a,b,N)+M*N] = -1


        if b == ymax and xmax==a:
            mat[j+2*M*N][be(a+1,b,N)] =1
            mat[j+2*M*N][be(a,b,N)] = -1
            mat[j+2*M*N][be(a,b+1,N)+M*N] = 1
            mat[j+2*M*N][be(a,b,N)+M*N] = -1
            #tar
            mat[j+2*M*N][be(a,b+1,N)+2*M*N] = -1
            mat[j+2*M*N][be(a,b,N)+2*M*N] = 1
                        #tar
            mat[j+2*M*N][be(a,b-2,N)+2*M*N] = -1
            mat[j+2*M*N][be(a,b,N)+2*M*N] = 2
            mat[j+2*M*N][be(a,b-1,N)+2*M*N] =-1

        # exter
        if b < ymax and xmin<a<xmax:
            mat[be(a,b,N)+2*M*N][be(a,b,N)+2*M*N] = 1
        # div nor
        if b == ymax and xmin<a<xmax:
            mat[j+2*M*N][be(a,b,N)+M*N] = -1
            mat[j+2*M*N][be(a,b+1,N)+M*N] = 1

            #tar
            mat[j+2*M*N][be(a,b,N)+2*M*N] = 1
            mat[j+2*M*N][be(a,b+1,N)+2*M*N] = -1
        if a==xmin and 0<b<ymax:
            mat[j+2*M*N][be(a-1,b,N)] = -1
            mat[j+2*M*N][be(a,b,N)] = 1
            #tar
            mat[j + 2 * M * N][be(a-1, b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(a, b, N)+2*M*N] = 1

        if a==xmax and 0<b<ymax:
            mat[j+2*M*N][be(a,b,N)] = -1
            mat[j+2*M*N][be(a+1,b,N)] = 1
            #tar

            mat[j + 2 * M * N][be(a, b, N)+2*M*N] = 1
            mat[j + 2 * M * N][be(a+1,b, N)+2*M*N] = -1


        if a==xmin and b==0:
            #tar
            mat[j + 2 * M * N][be(a-1, b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(a, b, N)+2*M*N] = 1

        if a==xmax and b==0:
            #tar

            mat[j + 2 * M * N][be(a, b, N)+2*M*N] = 1
            mat[j + 2 * M * N][be(a+1,b, N)+2*M*N] = -1

    return B,mat



a=matrixes(N,M,C,x,y,P1,P2,Q, xmin, xmax, ymax)[1]
b=matrixes(N,M,C,x,y,P1,P2,Q, xmin, xmax, ymax)[0]
#print(b)
#print(a)
print(np.linalg.det(a))
solution = np.linalg.solve(a, b)
ux = solution[:N*M]
uy = solution[N*M:2*N*M]
p = solution[2*N*M:]

u = np.array([np.sqrt(ux[i]**2 + uy[i]**2) for i in range(N*M)])

ab = np.linspace(0, L, N)
ord = np.linspace(0, l, M)
X, Y = np.meshgrid(ab, ord)
U=u.reshape(M,N)
UX=ux.reshape(M,N)
UY=uy.reshape(M,N)
P=p.reshape(M,N)

plt.pcolormesh(X, Y, U)
cbr=plt.colorbar( )
cbr.set_label("vitesse en m/s")
plt.quiver(X,Y,UX,UY)
plt.title("Allure de alure de sollution de l'equation Stokes avec déformation: Vitesse")
plt.xlabel("Distance en m")
plt.ylabel("Distance en m")
plt.show()

plt.pcolormesh(X, Y, P)
cbr=plt.colorbar( )
cbr.set_label("Pression en 10**(-2) Pa ")
plt.quiver(X,Y,UX,UY)
plt.title("Allure de alure de sollution de l'equation Stokes avec déformation: Pression")
plt.xlabel("Distance en m")
plt.ylabel("Distance en m")
plt.show()


"""def debit(u,N,M,l):
    L=[]
    for i in range (N):
        Um = 0
        for j in range (M):
            Um=Um+u[i+j*N]
        L.append(Um*l)
    return L"""
def debit(u,N,M,l,y):
    L=[]
    for i in range (N):
        Um = 0
        if i < xmin or i > xmax:
            for j in range (M):
                Um=Um+u[i+j*N]
            L.append(Um*l)
        else:
            for j in range (M):
                Um=Um+u[i+j*N]
            L.append(Um*(l-y*(ymax-1)))
    return L
D=debit(u,N,M,l,y)
plt.plot(ab, D, color='red')
plt.legend()
plt.title('Allure de debit')
plt.xlabel("abscice ")
plt.ylabel("Debit")
plt.show()
