
# -*- coding: utf-8 -*-

from PIL import Image
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### Parametres globaux
ngene = 1000 # Nombre de generations
taux = 0.3  # Taux de survie de chaque generation
mprob = 0.5 # Probabilite de mutation sur chaque gene
sCoef = 0.05 # coefficient de la fonction seuil
### Produit matriciel ( pour ne pas avoir a utiliser les array numpy )

def prod(A,B):
    if (len(A[0]) == len(B)):
        P=[[ 0 for _ in range(len(B[0]))] for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    P[i][j] += A[i][k]*B[k][j]
        return P
    else :
        return None

### Crossover entre deux matrices

def crossOver(A,B):
    if (len(A) == len(B) and len(A[0]) == len(B[0])):
        return [[ float(A[i][j]+B[i][j])/2 for j in range(len(A[0]))] for i in range(len(A))]
    else:
        return None

### Mutation sur une matrice

def muta(A,mrange,proba):
    for i in range(len(A)):
        for j in range(len(A[0])):
            if (rd.random()<proba):
                A[i][j] += mrange*(2*rd.random()-1)
    return A

### Fonctions de seuil

def seuil(A,a):
    return [[ np.exp(a*A[i][j])/(1+np.exp(a*A[i][j])) for j in range(len(A[0]))] for i in range(len(A))]

### Indice du maximum/minimum d'une liste

def indMax(L):
    ind=0
    for i in range(1,len(L)):
        if (L[i]>L[ind]):
            ind=i
    return ind

def indMin(L):
    ind=0
    for i in range(1,len(L)):
        if (L[i]<L[ind]):
            ind=i
    return ind

### Distance en norme 2 entre deux matrices de meme taille

def dist(A,B):
    if (len(A) == len(B) and len(A[0]) == len(B[0])):
        return sum([sum([ (A[i][j]-B[i][j])**2 for j in range(len(A[0]))]) for i in range(len(A))])**0.5
    else:
        return None
### fonction pour trancher sur le type de réponse
def s2(V):
    S=sum(V)
    return [np.exp(100*(v/S-0.95))/(1+np.exp(100*(v/S-0.95))) for v in V]

def reseau(L):
    mrange = 10  # Amplitude max des mutations ( degressif )
    npop = 20  # Taille de la population de synapses (pour l'algo genetique)
    ### Entrees M
    M=[]
    
    for path in L:
        with Image.open(path) as file :
            pix = file.load()
            Vect = []
            for i in range(6):
                for j in range(6):
                    if (pix[i,j] > 1):
                        Vect.append(1)
                    else:
                        Vect.append(0)
            M.append(Vect)
    
    ### Sorties S
    
    Sor=[[ 0 for _ in range(len(L))] for _ in range(len(L))]
    for i in range(len(L)):
        Sor[i][i] = 1
    
    ### Population de Synapse
    
    Syn = [[[ 20*rd.random()-1 for _ in range(len(L))] for _ in range(36)] for _ in range(npop)]
    
    ### Generations
    
    Erreurs = [[],[]]
    ok=0
    k=0
    while ok!=1:
        Score = []
        # Calcul des score de chaque synapse
        for i in range(npop):
            Score.append(dist(seuil(prod(M,Syn[i]),sCoef),Sor))
        # Extraction des meilleurs scores
        for _ in range(int(npop*(1-taux))):
            ind = indMax(Score)
            Syn.pop(ind)
            Score.pop(ind)
        # Repopulation
        NewSyn = []
        for _ in range(npop-len(Syn)):
            i,j=rd.randint(0,len(Syn)-1),rd.randint(0,len(Syn)-1)
            NewSyn.append(muta(crossOver(Syn[i],Syn[j]),mrange,mprob))
        Syn = Syn + NewSyn
        mrange *= 0.998 # Diminution de l'amplitude des mutations
    
        # Extraction reguliere de l'erreur pour affichage
        if (k%10 == 0):
            Erreurs[0].append(k)
            Erreurs[1].append(min(Score))
        if Erreurs[1][-1]<0.001:
            ok=1
        k+=1
    
    ### Affichage de l'evolution de l'erreur
    plt.title("Evolution de l'erreur")
    plt.plot(Erreurs[0], Erreurs[1])
    plt.show()
    
    ### Extraction du synapse optimal
    Score = []
    # Calcul des score de chaque synapse
    for i in range(npop):
        Score.append(dist(seuil(prod(M,Syn[i]),sCoef),Sor))
    ind = indMin(Score)
    Synapse = Syn[ind]
    return Synapse


### Concaténation des réseaux
V0123=['0.bmp','1.bmp','2.bmp','3.bmp']
V4567=['4.bmp','5.bmp','6.bmp','7.bmp']
V89PM=['8.bmp','9.bmp','+.bmp','-.bmp']
V3579=['3.bmp','5.bmp','7.bmp','9.bmp']
V2468=['2.bmp','4.bmp','6.bmp','8.bmp']

P0123=reseau(V0123)
P4567=reseau(V4567)
P89PM=reseau(V89PM)
P3579=reseau(V3579)
P2468=reseau(V2468)

def reseau1(V):
    Res=s2(seuil(prod(V,P0123),sCoef)[0])+s2(seuil(prod(V,P4567),sCoef)[0])+s2(seuil(prod(V,P89PM),sCoef)[0])
    return Res

def reseau2(V):
    Res=s2(seuil(prod(V,P3579),sCoef)[0])+s2(seuil(prod(V,P2468),sCoef)[0])
    return Res


### Réseau complet
Vcomp=['0.bmp','1.bmp','2.bmp','3.bmp','4.bmp','5.bmp','6.bmp','7.bmp','8.bmp','9.bmp','+.bmp','-.bmp']
Pcomp=reseau(Vcomp)

def reseaucomp(V):
    return seuil(prod(V,Pcomp),sCoef)[0]
### Ouverture du fichier Test

with Image.open('1.bmp') as file :
    pix = file.load()
    Vect = []
    for i in range(6):
        for j in range(6):
            if (pix[i,j] > 1):
                Vect.append(1)
            else:
                Vect.append(0)
Vect=[Vect]     # On travaille uniquement avec des tableaux

