# ------------------------------------------------------------------------------
# Regression logistique regularisee :
# Comme la regression logistique mais avec l'ajout d'un terme de regularisation
# a la fonction de cout. Cela evite l'overfitting en 'simplifiant' les donnees.
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

class logisticRegressionReg(object):
    """Modele de regression logistique avec regularisation"""

    def __init__(self, X, y):
        '''ROLE :
        PARAMETRES :
        RETOUR :
        '''
        self.X = X
        self.y = y

def plotFirst(X, y):
    pos = np.where(y == 1) # on recupere les indexes du tableau ou y == 1 sous forme de array
    neg = np.where(y == 0) # idem pour y == 0
    plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b') # on veut afficher un nuage de point ou les X de la 1ere et 2eme col sont == 1
    plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r') # idem mais == 0
    plt.xlabel('Score test 1') # ajoute une legende sur l'axe x
    plt.ylabel('Score test 2') # ajoute une legende sur l'axe y
    plt.legend(['Valide', 'Non valide']) # ajoute une legende
    plt.show() # affiche

data = np.loadtxt('data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

# plotFirst(X, y)


x1 = X[:,0]
x2 = X[:,1]

degree = 5

for i in range(1, degree):
    for j in range(0, i):
        print(i)
        print(j)
        x = np.zeros((np.shape(X[0]), 1))
        x = np.power(x1, i-j) * np.power(x2, j)
        X = np.c_[x, X]

        raw_input("coucou")
print(X)
