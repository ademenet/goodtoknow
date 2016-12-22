''' Nous allons recoder ici un programme de regression lineaire a plusieurs
variables.
'''

import numpy as np
import matplotlib.pyplot as plt

class linearRegression(object):
    """La regression lineraire permet de predire un resultat a partir d'un jeu
    de donnees et des resultats attendus (apprentissage supervise).
    Le but de la methode est de trouver les poids (thetas) permettant de modeliser
    une equation lineaire de la forme : h(x) = theta_0 * x_0 + theta_1 * x_1 +
    theta_2 * x_2 + ...
    Cette equation est notre hypothese, au sens statistique. Nous souhaitons donc
    trouver les thetas tel que le resultat de notre hypothese soit le plus proche
    des resultats attendus (y).
    Pour cela, nous utiliserons la fonction de cout (J) qui grace a sa derivee
    nous permettra de nous rapprocher du minimum, c'est a dire des thetas les
    plus proches. Donc, nous nous rapprocherons d'une estimation de plus en
    plus.
    """

    def __init__(self, X, y):
        self.m = X.shape[0] # nombre d'elements de mon dataset
        self.X = np.c_[np.ones(self.m), X] # ajoute une colonne de 1 au debut
        self.n = self.X.shape[1] # nombre de features
        self.y = y

    def linearHypothesis(self, x, theta):
        '''Calcul l'hypothese sous forme lineaire, quelquesoit le nombre de x et
        de theta.
        X : np.array de m * n
        theta : np.array de 1 * n
        Retourne h : np.array de m * 1
        '''
        return np.dot(theta.T, x)

    def computeCost(self, theta):
        '''Calcul l'erreur ou le cout en utilisant la "mean square error".
        y : ndarray de (m, 1)
        h : ndarray de (m, 1)
        J est un scalaire
        '''
        predict = self.linearHypothesis(self.X.T, theta)
        sq_err = (predict - self.y) ** 2 # calcul l'erreur carree pour chacun de nos elements
        # print(sq_err)
        # print self.m
        # print((1 / (2 * float(self.m))) * np.sum(sq_err))
        J = (1. / (2. * self.m)) * np.sum(sq_err) # calcul du cout/erreur
        # print J
        return J

    def gradientDescent(self, theta, alpha):
        '''Calcul le gradient, qui nous permet de mettre a jour les thetas pour
        s'approcher du minimum.
        alpha : float correspondant au learning rate
        Retourne un ndarray (n, 1)
        '''
        for i in range(self.m):
            h = self.linearHypothesis(self.X[i], theta)
            theta[0] = theta[0] - alpha * (1. / self.m) * np.sum(h - self.y[i])
            theta[1:] = theta[1:] - alpha * (1. / self.m) * np.sum((h - self.y[i]) * self.X[i, 1:]) # calcul le gradient pour chaque theta
        return theta

    def fit(self, iter, alpha):
        '''Entraine notre modele lineaire
        iter : int, nombre d'iteration a faire sur le dataset
        alpha : float, learning rate
        Retourne le ndarray (n, 1) contenant les thetas (poids) entraines
        '''
        theta = np.zeros((self.n, 1))
        J = np.zeros(iter * self.m)
        for i in range(iter):
            J[i] = self.computeCost(theta)
            theta = self.gradientDescent(theta, alpha)
        print J
        return theta, J

    def predict(self, x, theta):
        return self.linearHypothesis(x, theta)

def plot(X, y):
    plt.plot(X, y, 'ro')
    plt.show()

# Testons maintenant notre algorithme avec un exemple simple, les donnees sont
# dans data1.txt
# X et y sont sur une seule ligne et m colonnes
X, y = np.loadtxt('data1.txt', delimiter=',', unpack=True)
# X, y = np.loadtxt('data.csv', delimiter=',', unpack=True, skiprows=1, dtype=float)

model = linearRegression(X, y)

iter = 1500
alpha = 0.01
# print "Valeur initiale du cout :" + str(model.computeCost(X, y, theta))
theta, J = model.fit(iter, alpha)
print "Theta final:"
print(theta)
# print(J.T)
# plt.plot(J)
# plt.show()

predictions = np.zeros(X.shape[0])
i = 0
X_ = np.c_[np.ones(X.shape[0]), X]
for x in X_:
    predictions[i] = model.predict(x, theta)
    if i % 100 == 0:
        print str(i) + " / " + str(iter)
        print "y = " + str(y[i])
        print "predict = " + str(predictions[i])
    i += 1

# Plot les predictions et valeurs
plt.plot(X, y, 'ro')
plt.plot(X, predictions)
# Plot le cost
# plt.plot(cost, 'ro')
plt.show()
plt.waitforbuttonpress()
