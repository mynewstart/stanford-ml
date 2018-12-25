# coding: utf-8

import regression
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import  numpy as np

if __name__=='__main__':
    X,y=regression.loadDataSet('data/ex1.txt')

    m,n=X.shape
    X=np.concatenate((np.ones((m,1)),X),axis=1)

    rate=0.02
    maxLoop=1500
    epsilon=0.01

    result,timeConsumed=regression.bgd(rate,maxLoop,epsilon,X,y)

    theta,errors,thetas=result

    #绘制拟合曲线
    fittingFig=plt.figure()#figure对象 控制dpi、边界颜色、图形大小、和子区设置
    title='bgd: rate=%.2f,maxLoop=%d,epsilon=%.3f \n time: %ds'%(rate,maxLoop,epsilon,timeConsumed)
    ax=fittingFig.add_subplot(111,title=title)#将画布分成一行一列，将图像画在第一块上
    traingSet=ax.scatter(X[:,1].flatten().A[0],y[:,0].flatten().A[0])
    #X[:,1].flatten.A[0]将矩阵X的第2列（下标为1）展开成一个数组
    #ax.scatter()是绘制散点图，X,y为横纵坐标值


    xCopy=X.copy()
    xCopy.sort(0)
    yHat=xCopy*theta
    fittingLine,=ax.plot(xCopy[:,1],yHat,color='g')

    ax.set_xlabel('Population of City in 10,000s')
    ax.set_ylabel('Profit in $10,000s')

    plt.legend([traingSet,fittingLine],['Traing Set','Linear Regression'])#设置图例和文字的显示
    plt.show()

    #绘制误差曲线
    errorsFig=plt.figure()
    ax=errorsFig.add_subplot(111)
    #刻度设置
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.4f'))

    ax.plot(range(len(errors)),errors)
    ax.set_xlabel('Number of iterations')
    ax.set_ylabel('Cost J')

    plt.show()

    #绘制能量下降曲面
    size=100
    theta0Vals=np.linspace(-10,10,size)
    theta1Vals=np.linspace(-2,4,size)
    JVals=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            col=np.matrix([[theta0Vals[i]],[theta1Vals[j]]])
            JVals[i,j]=regression.J(col,X,y)

    #meshgrid将theta0Vals和theta1Vals扩展成行列相同的矩阵
    theta0Vals,theta1Vals=np.meshgrid(theta0Vals,theta1Vals)
    JVals=JVals.T
    contourSurf=plt.figure()
    ax=contourSurf.gca(projection='3d')

    #三维函数图像
    ax.plot_surface(theta0Vals, theta1Vals, JVals, rstride=2, cstride=2, alpha=0.3,
                    cmap=cm.rainbow, linewidth=0, antialiased=False)
    ax.plot(thetas[0], thetas[1], 'rx')
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    ax.set_zlabel(r'$J(\theta)$')

    plt.show()

    # 绘制能量轮廓
    contourFig = plt.figure()
    ax = contourFig.add_subplot(111)
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')

    #contour表示画等高线图，np.logspace是创建等比数列,从10^(-2)次方开始，到10^3
    CS = ax.contour(theta0Vals, theta1Vals, JVals, np.logspace(-2, 3, 20))
    plt.clabel(CS, inline=1, fontsize=10)

    # 绘制最优解
    ax.plot(theta[0, 0], theta[1, 0], 'rx', markersize=10, linewidth=2)

    # 绘制梯度下降过程
    ax.plot(thetas[0], thetas[1], 'rx', markersize=3, linewidth=1)
    ax.plot(thetas[0], thetas[1], 'r-')

    plt.show()