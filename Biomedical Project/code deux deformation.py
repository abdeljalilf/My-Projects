import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

N,M, L,l = 50, 26, 0.3, 0.1
x = L/N
y = l/M
C =  0.03
P1=130
P2=9
xmin, xmax, ymax = 24, 34, 6
xmin2, xmax2, ymin2 = 24,34 , 20

def be(i,j,N):
    return (j)*N + i
def re(k,N):
    return (k%N,k//N)
#Q=[(0,2),(0,4),(2,2),(2,4)]


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
def deformer2(Xmin,Xmax,Ymin):
    L=[]
    x=Xmin
    y=Ymin
    while Xmin <= x <= Xmax and M-1 >= y  :

        L.append((x,y))
        if x==Xmax :
            x=Xmin
            y+=1
            continue
        x+=1

    return L



Q = deformer(xmin, xmax, ymax)
Q2=deformer2(xmin2, xmax2, ymin2)
#print(Q)
def matrixes(N,M,C,x,y,P1,P2,Q,Q2, xmin, xmax, ymax, xmin2, xmax2, ymin2):
    Y=y**2
    X=x**2
    B= np.zeros((3*N*M,1))
    mat = np.zeros((3*N*M,3*N*M))
    for j in range(N*M):
        a=re(j,N)[0]
        b=re(j,N)[1]
        if a!=0 and a!=N-1 and b!=0 and b!=M-1 and re(j,N) not in Q and re(j,N) not in Q2 :
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
            if  a!=0 and a!= N-1 and re(j,N) not in Q2 :
                mat[j+2*M*N][be(a,b-1,N)+M*N] = -1
                mat[j+2*M*N][be(a,b,N)+M*N] = 1
                #tar
                mat[j+2*M*N][be(a,b-2,N)+2*M*N] = 1
                mat[j+2*M*N][be(a,b,N)+2*M*N] = 1
                mat[j+2*M*N][be(a,b-1,N)+2*M*N] = -2

        #Q

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

        # exter
        if b < ymax and xmin<a<xmax:
            mat[be(a,b,N)+2*M*N][be(a,b,N)+2*M*N] = 1
        # div nor
        if b == ymax and xmin<a<xmax:
            mat[j+2*M*N][be(a,b,N)+M*N] = -1
            mat[j+2*M*N][be(a,b+1,N)+M*N] = 1
            #tar
            mat[j+2*M*N][be(a,b-2,N)+2*M*N] = 1
            mat[j+2*M*N][be(a,b,N)+2*M*N] = 1
            mat[j+2*M*N][be(a,b-1,N)+2*M*N] = -2

        if a==xmin and 0<b<ymax:
            mat[j+2*M*N][be(a-1,b,N)] = -1
            mat[j+2*M*N][be(a,b,N)] = 1
            mat[j + 2 * M * N][be(a-1, b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(a, b, N)+2*M*N] = 1
            mat[j + 2 * M * N][be(xmax+1,b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(xmax, b, N)+2*M*N] = 1

        if a==xmax and 0<b<ymax:
            mat[j+2*M*N][be(a,b,N)] = -1
            mat[j+2*M*N][be(a+1,b,N)] = 1

        if a==xmin and b==0:
            #mat[be(a,b,N)+2*M*N][be(a,b,N)+2*M*N] = 1
            #tar
            mat[j + 2 * M * N][be(a-1, b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(a, b, N)+2*M*N] = 1
            mat[j + 2 * M * N][be(xmax+1,b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(xmax, b, N)+2*M*N] = 1
        if a==xmax and b==0:
            #mat[be(a,b,N)+2*M*N][be(a,b,N)+2*M*N] = 1
            #tar
            mat[j + 2 * M * N][be(a-1, b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(a, b, N)+2*M*N] = 1
            mat[j + 2 * M * N][be(xmax+1,b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(xmax, b, N)+2*M*N] = 1

        #Q2


        #U=0
        if re(j,N) in Q2:
            mat[j][be(a,b,N)] = 1
            mat[j+M*N][be(a,b,N)+M*N] = 1
         # div
        if b == ymin2 and xmin2==a:
            mat[j+2*M*N][be(a,b,N)] =1
            mat[j+2*M*N][be(a-1,b,N)] = -1
            mat[j+2*M*N][be(a,b,N)+M*N] = 1
            mat[j+2*M*N][be(a,b-1,N)+M*N] = -1
        if b == ymin2 and xmax2 ==a:
            mat[j+2*M*N][be(a+1,b,N)] =1
            mat[j+2*M*N][be(a,b,N)] = -1
            mat[j+2*M*N][be(a,b,N)+M*N] = 1
            mat[j+2*M*N][be(a,b-1,N)+M*N] = -1


         #

        """ if re(j,N)==(2,1):
            mat[j+2*M*N][be(a-1,b,N)+2*M*N] = -1
            mat[j+2*M*N][be(a,b,N)+2*M*N] = 1"""
        # exter
        if b > ymin2 and xmin2 < a < xmax2:
            mat[be(a,b,N)+2*M*N][be(a,b,N)+2*M*N] = 1
        # div nor
        if b == ymin2 and xmin2 < a < xmax2:
            mat[j+2*M*N][be(a,b-1,N)+M*N] = -1
            mat[j+2*M*N][be(a,b,N)+M*N] = 1
            #tar
            mat[j+2*M*N][be(a,b-2,N)+2*M*N] = 1
            mat[j+2*M*N][be(a,b,N)+2*M*N] = 1
            mat[j+2*M*N][be(a,b-1,N)+2*M*N] = -2

            '''mat[j + 2 * M * N][be(a, b, N) + M * N] = -1
            mat[j + 2 * M * N][be(a, b + 1, N) + M * N] = 1
            mat[j + 2 * M * N][be(a, M-1, N) + M * N] = 1
            mat[j + 2 * M * N][be(a, M - 2, N) + M * N] = -1'''

        if a==xmin2 and b>ymin2:
            mat[j+2*M*N][be(a-1,b,N)] = -1
            mat[j+2*M*N][be(a,b,N)] = 1
            mat[j + 2 * M * N][be(a-1, b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(a, b, N)+2*M*N] = 1
            mat[j + 2 * M * N][be(xmax2+1,b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(xmax2, b, N)+2*M*N] = 1

            """#tar_2

            mat[j+2*M*N][be(a-1,b-2,N)+2*M*N] = 1
            mat[j+2*M*N][be(a-1,b,N)+2*M*N] = 1
            mat[j+2*M*N][be(a-1,b-1,N)+2*M*N] = -2 """


        if a==xmax2 and ymin2 < b:
            mat[j+2*M*N][be(a,b,N)] = -1
            mat[j+2*M*N][be(a+1,b,N)] = 1

        if a==xmin2 and b==M-1:
            #mat[be(a,b,N)+2*M*N][be(a,b,N)+2*M*N] = 1
            #tar
            mat[j + 2 * M * N][be(a-1, b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(a, b, N)+2*M*N] = 1
            mat[j + 2 * M * N][be(xmax2+1,b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(xmax2, b, N)+2*M*N] = 1
        if a==xmax2 and b==M-1:
            #mat[be(a,b,N)+2*M*N][be(a,b,N)+2*M*N] = 1
            #tar
            mat[j + 2 * M * N][be(a-1, b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(a, b, N)+2*M*N] = 1
            mat[j + 2 * M * N][be(xmax+1,b, N)+2*M*N] = -1
            mat[j + 2 * M * N][be(xmax, b, N)+2*M*N] = 1



    return B,mat

a=matrixes(N,M,C,x,y,P1,P2,Q,Q2, xmin, xmax, ymax, xmin2, xmax2, ymin2)[1]
b=matrixes(N,M,C,x,y,P1,P2,Q,Q2, xmin, xmax, ymax, xmin2, xmax2, ymin2)[0]
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
TU = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        TU[i][j] = [N*i+j][0]
T = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        T[i][j] = p[N*i+j][0]
TX = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        TX[i][j] = ux[N*i+j][0]
TY = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        TY[i][j] = uy[N*i+j][0]
plt.pcolormesh(X, Y, T)

plt.colorbar()
plt.quiver(X,Y,TX,TY)
plt.title('Allure de la vitesse')
plt.show()

TX = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        TX[i][j] = ux[N*i+j][0]
TY = np.zeros((M, N))
for i in range(M):
    for j in range(N):
        TY[i][j] = uy[N*i+j][0]
'''plt.pcolormesh(X, Y, TX)
plt.colorbar()
plt.title('Allure de Ux')
#plt.show()
plt.pcolormesh(X, Y, TY)
plt.colorbar()
plt.title('Allure de Uy')
#plt.show()
mat = a
def detect(mat):
    lou = len(mat)
    colones = []
    for j in range(lou):
        colones.append([mat[i][j] for i in range(lou)])
    return colones
n, s = [], []
for c in detect(mat):
    for k in c:
        if k != 0:
            n.append(detect(mat).index(c))
for k in n:
    if k not in s:
        s.append(k)


print(s)
print(detect(a)[93])
print(detect(a)[91])
print(a[14])'''