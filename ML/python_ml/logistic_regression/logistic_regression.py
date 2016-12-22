# ------------------------------------------------------------------------------
# Regression logistique :
# Nous allons faire un algorithme de regression logistique en nous basant sur
# les donnees du cours de Andrew Ng sur Coursera et en s'aidant de la fonction
# de minimisation fmin_bfgs de scipy
# ------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

class logisticRegression(object):
    '''Cette classe contient un certain nombre de methodes qui vont nous
    permettre d'effectuer notre logistic regression
    '''

    def __init__(self, X, y):
        '''ROLE :
        Constructeur

        PARAMETRES :
        X : matrice des exemples
        y : matrice de labels
        m : nombre d'exemples (nombre de lignes de X)
        n : nombre de features, incluant les bias - colonne de 1 (nombre de
            colonnes de X)

        RETOUR :
        '''
        # Normalisation des donnees
        for col in range(X.shape[1]):
            X[:, col] -= X[:, col].min() # Ou X[:, col] -= X[:, col].mean()
            X[:, col] /= X[:, col].std()
        # On ajoute une colonne de 1 (le bias) a notre matrice X. On se retrouve
        # avec une matrice de dimension X(m, n + 1) :
        self.X = np.c_[np.ones(X.shape[0]), X]
        self.m, self.n = self.X.shape
        # Convertit y en un tableau/matrice a 2 dimensions propre, afin de
        # faciliter les calculs matriciels par la suite :
        if len(y.shape) == 1:
            y.shape = (y.shape[0],1)
        self.y = y

    def sigmoid(self, z):
        '''ROLE :
        Calcule la fonction sigmoid de z

        PARAMETRES :
        z : matrice, vecteur ou scalaire

        RETOUR :
        La valeur de la fonction sigmoid
        '''
        return 1.0 / (1.0 + np.exp(-z))

    # def softmax(self, z):
    #     '''ROLE :
    #     Calcule la fonction Softmax de z
    #
    #     PARAMETRES :
    #     z : vecteur de dimension (m, 1)
    #
    #     RETOUR :
    #     Retourne un vecteur de dimension (m, 1)
    #     '''
    #     tmp = np.exp(z)
    #     return tmp / np.sum(tmp, axis=1)

    def computeCost(self, theta):
        '''ROLE :
        Calcule la fonction de cout, J

        PARAMETRES :
        theta : les poids que nous devons entrainer, vecteur

        RETOUR :
        Retourne une valeur correspondante a l'erreur pour un theta donne
        '''
        h = self.sigmoid(np.dot(self.X, theta))
        return np.sum(-self.y * np.log(h) - (1.0 - self.y) * np.log(1.0 - h)) / self.m

    def computeGradient(self, theta):
        '''ROLE :
        Calcule le gradient avec iteration
        ---> A faire : trouver la formule vectoriser

        PARAMETRES :
        theta : les poids que nous devons entrainer, vecteur

        RETOUR :
        Le gradient sous forme d'un vecteur de dimension (n, 1), puisqu'il
        correspond a l'ecart/erreur entre notre estimation (h calcule avec nos
        thetas) et le resultat attendu
        '''
        grad = np.zeros(self.n)
        error = self.sigmoid(np.dot(self.X, theta)) - self.y
        for i in range(self.n):
            term = np.multiply(error.T, self.X[:,i])
            grad[i] = np.sum(term) / self.m
        return grad

    def fit(self, iter=400):
        '''ROLE :
        Entraine notre algorithme afin de minimiser les parametres thetas

        PARAMETRES :
        iter : nombre d'iterations (batch)

        RETOUR :
        Nos parametres contenus dans theta, vecteur
        '''
        # Initialise theta avec des zeros :
        theta_initial = np.zeros((self.n, 1))
        # Calcul du cout initial :
        cost = self.computeCost(theta_initial)
        # Calcul du gradient initial :
        grad = self.computeGradient(theta_initial)

        print 'Cout initial : {0} \nGradient initial : {1}'.format(cost, grad)

        # La fonction fmin_bfgs ne marche pas :'-(
        # theta = opt.fmin_bfgs(f=self.computeCost, x0=theta_initial,
        #                       fprime=self.computeGradient, maxiter=400)

        theta = opt.minimize(fun = self.computeCost,
                             x0 = theta_initial,
                             method = 'TNC',
                             jac = self.computeGradient)
        return theta.x

    def predict(self, x, theta):
        predict = self.sigmoid(np.dot(x, theta))
        if predict >= 0:
            y = 1
        else:
            y = 0
        return y

def plotFirst(X, y):
    pos = np.where(y == 1) # on recupere les indexes du tableau ou y == 1 sous forme de array
    neg = np.where(y == 0) # idem pour y == 0
    plt.scatter(X[pos, 0], X[pos, 1], marker='o', c='b') # on veut afficher un nuage de point ou les X de la 1ere et 2eme col sont == 1
    plt.scatter(X[neg, 0], X[neg, 1], marker='x', c='r') # idem mais == 0
    plt.xlabel('Exam 1 score') # ajoute une legende sur l'axe x
    plt.ylabel('Exam 2 score') # ajoute une legende sur l'axe y
    plt.legend(['Not Admitted', 'Admitted']) # ajoute une legende
    plt.show() # affiche

data = np.loadtxt('data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]

plotFirst(X, y)

model = logisticRegression(X, y)

theta = model.fit()
print "Theta final : {0}".format(theta)

cost = model.computeCost(theta)
print "Cost final : {0}".format(cost)

pred = model.predict([1, 0.9, 0.01], theta)
print "Prediction : {0}".format(pred)

