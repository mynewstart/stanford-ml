
# coding: utf-8
import regression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from mpl_toolkits.mplot3d import axes3d

if __name__=="__main__":
    srcX,y=regression.loadDataSet('data/houses.txt')

    m,n=srcX.shape
    #concatenate是数组拼接函数，axix=0是按照行拼接，axis=1是按照列进行
    srcX = regression.standarize(srcX)
    X=np.concatenate((np.ones((m,1)),srcX),axis=1)

    rate=0.01
    maxLoop=1000
    epsilon=1

    result,timeConsumed=regression.bgd(rate,maxLoop,epsilon,X,y)
    theta, errors, thetas = result
    print(theta)
    print(errors)
    print(thetas)
    #打印拟合曲线
    #控制dpi、边界颜色、图形大小、和子区设置
    fittingFig=plt.figure()
    title='bgd: rate=%.2f, maxLoop=%d, epsilon=%.3f \n time: %ds'%(rate,maxLoop,epsilon,timeConsumed)
    #将画布分成一行一列，将图像画在第一块上
    ax=fittingFig.add_subplot(111,title=title)
    trainingSet=ax.scatter(X[:,1].flatten().A[0],y[:,0].flatten().A[0])

    xCopy=X.copy()
    xCopy.sort(0)
    yHat=xCopy*theta
    fittingLine,=ax.plot(xCopy[:,1],yHat,color='g')

    ax.set_xlabel('Population of City in 10,000s')
    ax.set_ylabel('profit in $10,1000s')

    plt.legend([trainingSet,fittingLine],['Training Set','Liner Regression'])

    plt.show()

    """
    #绘制能量函数的轮廓
    theta1Vals = np.linspace(min(thetas[1]), max(thetas[1]), 100)
    theta2Vals = np.linspace(min(thetas[2]), max(thetas[2]), 100)
    JVals = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            theta = np.matrix([[0], [theta1Vals[i]], [theta2Vals[j]] ])
            JVals[i, j] = regression.J(theta, X, y)
    contourFig = plt.figure()
    ax = contourFig.add_subplot(111)
    ax.contour(theta1Vals, theta2Vals, JVals, np.logspace(-2, 3, 20))

    plt.show()

    # 打印误差曲线
    errorsFig = plt.figure()
    ax = errorsFig.add_subplot(111)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))

    ax.plot(range(len(errors)), errors)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost J')

    plt.show()
"""


