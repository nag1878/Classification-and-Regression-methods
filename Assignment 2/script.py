import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi,exp
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    X = np.asarray(X)
    y=np.asarray(y)
    choicelist = [X]
    select_array = []
    #average_mat = (np.empty([X.shape[1],0]))
    #print(average_mat)
    #print(average_mat.shape)
    i=1
    for i in range (1,(len(np.unique(y))+1)):
        condlist = [y==i]
        new_row = np.select(condlist,choicelist)
        #print("before")
        #print(new_row.shape)
        new_row= new_row[~(new_row==0).all(1)] 
        #print(new_row.shape)
        #print(final_array.shape)   
        mean_new = (np.mean(new_row, axis=0))   
        select_array = np.append(select_array,mean_new,axis=0)
        #print(np.unique(y).shape[0])
    select_array = np.reshape(np.transpose(select_array), ((np.unique(y).shape[0], X.shape[1])))
    #print(select_array)
    #means_array = np.asarray(np.split(select_array,len(np.unique(y)),axis=0))
    means =(np.transpose(select_array))    
    covmat=np.cov(np.transpose(X),bias=True)
    #print(means.shape, covmat.shape)
    #print(means)
    #print(covmat)
    #
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    
    X = np.asarray(X)
    y=np.asarray(y)
    choicelist = [X]
    select_array = []
    cov_array_temp = []
    covmats=np.zeros((X.shape[1], X.shape[1]))
    #print(covmats)
    #average_mat = (np.empty([X.shape[1],0]))
    #print(average_mat)
    #print(average_mat.shape)
    i=1
    for i in range (1,(len(np.unique(y))+1)):
        condlist = [y==i]
        new_row = np.select(condlist,choicelist)
        #print("before")
        #print(new_row.shape)
        new_row= new_row[~(new_row==0).all(1)] 
        #print(new_row.shape)
        #print(final_array.shape)
        #print(new_row.shape)
        cov_array_temp = np.append(cov_array_temp, np.cov(np.transpose(new_row)))
        #print(cov_array_temp)  
        #print("check") 
        mean_new = (np.mean(new_row, axis=0))   
        select_array = np.append(select_array,mean_new,axis=0)
        #print(np.unique(y).shape[0])
    temp = np.reshape(cov_array_temp, (np.unique(y).shape[0],4))
    for row in temp:
        covmats =np.vstack((covmats, np.reshape(row, (X.shape[1], X.shape[1])))) 
    covmats = np.delete(covmats, [0,1], axis=0)
    covmats = np.vsplit(covmats,np.unique(y).shape[0])
    #print(covmats)
    select_array = np.reshape(np.transpose(select_array), ((np.unique(y).shape[0], X.shape[1])))
    #print(select_array)
    #means_array = np.asarray(np.split(select_array,len(np.unique(y)),axis=0))
    means =(np.transpose(select_array)) 
    
    #print(means,covmats)
    
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    i=0
    j=0
    
    class_calculate = np.zeros([ytest.shape[0],ytest.shape[1]])
    #print(class_calculate.shape)
    means = np.transpose(means)
    #print(means.shape)
    for i in range(0,(Xtest.shape[0])):
        prod_final_mat=[]    
        for j in range(0,means.shape[0]):    
            inv = (np.linalg.inv(covmat))
            sub_temp =np.subtract(Xtest[i], means[j])
            #print("sub_temp"+str(sub_temp.shape))
            prod_temp = np.dot(sub_temp,inv)
            prod_final = np.dot(np.transpose(sub_temp), prod_temp)
            prod_final_mat = np.append(prod_final_mat,prod_final)
        class_calculate = (np.append(class_calculate,(np.argmin(prod_final_mat)+1)))
    class_calculate = class_calculate[~(class_calculate==0)]
    class_calculate = np.reshape(np.asarray(class_calculate),(ytest.shape[0], ytest.shape[1]))
    #print(class_calculate.shape)
    #print(ytest.shape)
    x = (class_calculate == ytest)
    ypred = class_calculate
    acc = (np.sum(x))
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    #print(covmats)
    i=0
    j=0
    
    class_calculate = np.zeros([ytest.shape[0],ytest.shape[1]])
    #print(class_calculate.shape)
    means = np.transpose(means)
    #print(means.shape)
    for i in range(0,(Xtest.shape[0])):
        prod_final_mat=[]    
        for j in range(0,means.shape[0]):    
            inv = (np.linalg.inv(covmats[j]))
            sub_temp =np.subtract(Xtest[i], means[j])
            #print("sub_temp"+str(sub_temp.shape))
            prod_temp = np.dot(sub_temp,inv)
            prod_final = np.dot(np.transpose(sub_temp), prod_temp)
            #print(prod_final)
            prod_final = (exp(-0.5*prod_final))/(np.linalg.det(covmats[j]))**(.5)
            prod_final_mat = np.append(prod_final_mat,prod_final)
        class_calculate = (np.append(class_calculate,(np.argmax(prod_final_mat)+1)))
    class_calculate = class_calculate[~(class_calculate==0)]
    class_calculate = np.reshape(np.asarray(class_calculate),(ytest.shape[0], ytest.shape[1]))
    #print(class_calculate.shape)
    #print(ytest.shape)
    x = (class_calculate == ytest)
    #print(np.sum(x))
    
    acc=np.sum(x)
    ypred=class_calculate
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
	
    # IMPLEMENT THIS METHOD
    part_inverse = np.dot(np.transpose(X),X)
    w1 = np.dot(inv(part_inverse),np.transpose(X))
    w = np.dot(w1,y)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD
    id = np.identity(X.shape[1])
    term_one = np.multiply(lambd,id)
    term_two = np.dot(np.transpose(X),X)
    part_one = inv(np.add(term_one,term_two))
    w = np.dot(np.dot(part_one,np.transpose(X)),y)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    w_transpose = np.transpose(w)
    sum_value = 0
    mse = 0
    
    # IMPLEMENT THIS METHOD
    N = ytest.shape[0]
    w_transpose = np.transpose(w)
    sum_value = 0

    for i in range(0,ytest.shape[0]):
    	part_one = ytest[i][0] - np.dot(w_transpose,Xtest[i])
    	sum_value = sum_value + np.square(part_one)

    mse = sum_value/N
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD
    term_1 = y[:,0] - np.dot(X,w)
    error =  0.5 *  (  (  np.dot(term_1.transpose() , term_1)  ) + (  lambd * np.dot(w.transpose(),w) )   )   
    
    term_2 = np.dot(X,w) - y[:,0]    
    error_grad = np.dot( term_2.transpose(), X ) + (lambd*w)
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1)) 
	
    # IMPLEMENT THIS METHOD
    Xd = np.zeros((x.shape[0],p+1))
    #x = np.reshape(x,(x.shape[0],1))
    for i in range(0,p+1):
    	#p_e = np.array(x**i)
    	#print("shape of ele",p_e.shape)
    	#Xd = np.concatenate((Xd,p_e),1)
    	#print("shape of Xd", Xd.shape)
    	Xd[:,i] = np.power(x,i)
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('D:\MS\ML\Assignment 2\sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('D:\MS\ML\Assignment 2\sample.pickle','rb'),encoding = 'latin1')
 #LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
## QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = 0 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()
