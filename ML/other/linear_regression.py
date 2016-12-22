import matplotlib
import numpy as np

def getData(fileName):
    dataSet = np.loadtxt(fileName, delimiter=',')
    return dataSet

def computeCost(X, y, theta, m):
    prediction = X * theta
    square_error = (prediction - y) ** 2
    J = 1 / (2 * m) * np.sum(square_error)
    return J

def gradientDescent(X, y, theta, m, alpha, iters):
    x = X[:,1]
    for i in xrange(1, iters):
        hypothesis = theta[0] + theta[1] * x
        # print hypothesis - y
        theta_0 = theta[0] + alpha * (1 / m) * np.sum(hypothesis - y)
        # print theta_0
        theta_1 = theta[0] + alpha * (1 / m) * np.sum(np.multiply((hypothesis - y), x))
        # print theta_1
        theta[0] = theta_0
        # print theta_0
        theta[1] = theta_1
        # print theta_1
    return theta

def main():
    dataSet = getData('ex1data1.txt')
    X_temp = dataSet[:,0]
    y = dataSet[:,1]
    m = X_temp.size
    theta = np.zeros((2,1))
    cost = computeCost(X_temp, y, theta, m)
    print cost
    X = np.c_[np.ones(m), X_temp]
    theta = gradientDescent(X, y, theta, m, 0.01, 1000)
    print theta

main()