# coding: utf-8

import numpy as np
import  matplotlib as plt
import time

def exeTime(func):
    """可变参数args和关键字参数args2"""
    def newFunc(*args,**args2):
        t0=time.time()
        back=func(*args,**args2)
        return back,time.time()-t0
    return newFunc

def loadDataSet(filename):
    """特征数量"""
    numFeat=len(open(filename).readline().split('\t')) - 1
    X=[]
    y=[]
    file = open(filename)
    for line in file.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))

        X.append(lineArr)
        y.append(float(curLine[-1]))

    return np.mat(X),np.mat(y).T

#特征标准化
def standarize(X):
    m,n=X.shape
    for j in range(n):
        features=X[:,j]
        meanVal=features.mean(axis=0)
        std=features.std(axis=0)  #std是标准差
        if std!=0:
            X[:,j]=(features-meanVal)/std
        else:
            X[:,j]=0
    return X

#归一化
def normalize(X):
    for j in range(n):
        features=X[:,j]
        minVal=features.min(axis=0)
        maxVal=features.max(axis=0)
        diff=maxVal-minVal
        if diff!=0:
            X[:,j]=(features-minVal)/diff
        else:
            X[:,j]=0

    return X


def h(theta,x):
    return (theta.T*x)[0,0]

def J(theta,X,y):
    """theta是n*1的矩阵，X是m*n的矩阵,y是m*1的矩阵"""
    m=len(X)
    return (X*theta-y).T*(X*theta-y)/(2*m)

@exeTime
def bgd(rate, maxLoop, epsilon, X, y):
    """批量梯度下降法
    Args:
        rate 学习率
        maxLoop 最大迭代次数
        epsilon 收敛精度
        X 样本矩阵
        y 标签矩阵
    Returns:
        (theta, errors, thetas), timeConsumed
    """
    m,n = X.shape
    # 初始化theta
    theta = np.zeros((n,1))
    count = 0
    converged = False
    error = float('inf')
    errors = []
    thetas = {}
    for j in range(n):
        thetas[j] = [theta[j,0]]
    while count<=maxLoop:
        if(converged):
            break
        count = count + 1
        for j in range(n):
            deriv = (y-X*theta).T*X[:, j]/m
            theta[j,0] = theta[j,0]+rate*deriv
            thetas[j].append(theta[j,0])
        error = J(theta, X, y)
        errors.append(error[0,0])
        # 如果已经收敛
        if(error < epsilon):
            converged = True
    return theta,errors,thetas


@exeTime
def sgd(rate, maxLoop, epsilon, X, y):
    """随机梯度下降法
    Args:
        rate 学习率
        maxLoop 最大迭代次数
        epsilon 收敛精度
        X 样本矩阵
        y 标签矩阵
    Returns:
        (theta, error, thetas), timeConsumed
    """
    m,n = X.shape
    # 初始化theta
    theta = np.zeros((n,1))
    count = 0
    converged = False
    error = float('inf')
    errors = []
    thetas = {}
    for j in range(n):
        thetas[j] = [theta[j,0]]
    while count <= maxLoop:
        if converged:
            break
        count = count + 1
        errors.append(float('inf'))
        for i in range(m):
            if converged:
                break
            diff = y[i,0]-h(theta, X[i].T)
            for j in range(n):
                theta[j,0] = theta[j,0] + rate*diff*X[i, j]
                thetas[j].append(theta[j,0])
            error = J(theta, X, y)
            errors[-1] = error[0,0]
            # 如果已经收敛
            if(error < epsilon):
                converged = True
    return theta, errors, thetas

def JLwr(theta, X, y, x, c):
    """局部加权线性回归的代价函数计算式
    Args:
        theta 相关系数矩阵
        X 样本集矩阵
        y 标签集矩阵
        x 待预测输入
        c tau
    Returns:
        预测代价
    """
    m,n = X.shape
    summerize = 0
    for i in range(m):
        diff = (X[i]-x)*(X[i]-x).T
        w = np.exp(-diff/(2*c*c))
        predictDiff = np.power(y[i] - X[i]*theta,2)
        summerize = summerize + w*predictDiff
    return summerize


def JLwr(theta,X,y,x,c):
    m,n=X.shape
    summerize=0
    for i in range(m):
        diff=(X[i]-x)*(X[i]-x).T
        w=np.exp(-diff/(2*c*c))
        predictDiff=np.power(y[i]-X[i]*theta,2)
        summerize=summerize+w*predictDiff

    return summerize

def lwr(rate,maxLoop,epsilon,X,y,x,c=1):
    """

    :param rate:
    :param maxLoop:
    :param epsilon:
    :param X:
    :param y:
    :param x: 带预测向量
    :param c: tau值
    :return:
    """
    m,n=X.shape
    theta=np.zeros((n,1))
    count=0
    converged=False
    error=float('inf')
    errors=[]
    thetas={}
    for j in range(n):
        thetas[j]=[theta[j,0]]

    #bgd
    while count<=maxLoop:
        if(converged):
            break
        count+=1
        for j in range(n):
            deriv=(y-X*theta).T*X[:,j]/m
            theta[j,0]=theta[j,0]+rate*deriv
            thetas[j].append(theta[j,0])
        #计算局部加权函数的代价函数
        error=JLwr(theta,X,y,x,c)
        errors.append(error[0, 0])
        # 如果已经收敛
        if (error < epsilon):
            converged = True
    return theta, errors, thetas