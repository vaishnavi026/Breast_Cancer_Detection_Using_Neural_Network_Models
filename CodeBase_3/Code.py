import csv
import math
import numpy as np
import time
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier

def normalize(x, min, max):
    return (2*(x-min)/(max-min))-1

with open('data2.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    X=[]
    Y=[]
    Z=[]
    for row in readCSV:
        Y.append(row[9])
        Y=list(map(int,Y))
        con = [int(i) for i in row[0:9]]
        X.append(np.array(con))
        con1 = [int(j) for j in row[0:10]]
        Z.append(np.array(con1))
    X=np.array(X)
    Z=np.array(Z)
    for i in range(683):
        if Y[i]==2:                         #Binary classification
            Y[i]=0
            Z[i][9]=0
        else:
            Y[i]=1
            Z[i][9] = 1
    X= (2/9)*X-11/9
    #X=normalize(X,1, 10)

    Z1 = []
    Z2 = []
    Z3 = []
    for i in range(683):
        Z3.append(Z[i, :9])
        if Z[i, 9] == 0:
            Z1.append(Z[i, :9])
        else:
            Z2.append(Z[i, :9])
    Z1 = np.array(Z1)
    Z1 = 2 / 9 * Z1 - 11 / 9
    #Z1=normalize(Z1,1,10)
    Z2 = np.array(Z2)
    Z2 = 2 / 9 * Z2 - 11 / 9
    #Z2 = normalize(Z2, 1, 10)
    Z3 = np.array(Z3)
    Z3 = 2 / 9 * Z3 - 11 / 9
    #Z3 = normalize(Z3, 1, 10)

    pca_nrm = PCA(n_components=2)                                   #to reduce dimension to 2 for plotting in 2D
    X_nrm = pca_nrm.fit_transform(Z3)
    plt.scatter(X_nrm[:, 0], X_nrm[:, 1])
    plt.show()
    X1 = []
    X2 = []
    for i in range(683):
        # Z3.append(Z[i,:9])
        if Z[i, 9] == 0:
            X1.append(X_nrm[i, :2])
        else:
            X2.append(X_nrm[i, :2])
    X1 = np.array(X1)
    X2 = np.array(X2)
    plt.scatter(X1[:, 0], X1[:, 1], color='r')
    plt.scatter(X2[:, 0], X2[:, 1], color='g')
    plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_der(z):
    return sigmoid(z)*(1-sigmoid(z))

class NeuralNet:
    def __init__(self, hidden_neurons):
        self.hn = hidden_neurons
        self.weights0 = np.random.randn(self.hn, 9)                     # weights for hidden layer
        self.biases0 = np.random.randn(self.hn, 1)                      # biases for hidden layer
        self.weights1 = np.random.randn(1, self.hn)                     # weights for output w1= 1x9
        self.biases1 = np.random.randn(1, 1)                            # biases for output b1 = 1x1

    def train(self, inputs, outputs, alpha, epochs, error_fun,hn):
        outputs = np.array([outputs]).T
        op_err1 = []
        ep = []
        for i in range(epochs):
            op_err = 0
            for j in range(len(inputs)):
                z = np.dot(self.weights0, (np.array([inputs[j]])).T) + self.biases0
                foz = sigmoid(z)                                         # output of hidden layer # foz = 9x1
                y = np.dot(self.weights1, foz) + self.biases1  # y= 1x1
                foy = sigmoid(y)                                         # final output # foy = 1x1
                k1 = foy[0][0]                                           #obtained output
                k2 = outputs[j][0]                                       #target output
                op_err = k2 - k1                                         # error in output

                if error_fun == 0:
                    delta = op_err * sigmoid_der(y)[0][0]                # mse error function
                else:
                    delta = op_err                                       # cross entropy error function

                self.weights1 = self.weights1 + alpha * delta * z.T      # updated weight1
                self.biases1 = self.biases1 + alpha * delta              # updates biases1
                delta2 = delta * sigmoid_der(z)
                self.weights0 = self.weights0 + alpha * (np.dot(delta2, np.array([inputs[j]])))  # updated weight0
                self.biases0 = self.biases0 + alpha * delta2  # updated biases0
            op_err1.append(op_err)
            ep.append(i)
        np.array(op_err1)
        np.array(ep)

        fig, ax = plt.subplots()
        ax.plot(ep, op_err1)
        ax.set(xlabel='Epoch', ylabel='Error', title='Change in Error with Epochs for alpha '+str(alpha)+ ' hidden node'+ str(hn))
        ax.grid()
        plt.show()

    def predict_1(self, inputs):
        inputs=normalize(np.array(inputs),1,10)
        k1 = []
        np.array(inputs)

        for j in range(len(inputs)):
            z = np.dot(self.weights0, (np.array([inputs[j]])).T) + self.biases0
            foz = sigmoid(z)                                     # output of hidden layer
            y = np.dot(self.weights1, foz) + self.biases1
            foy = sigmoid(y)                                     # final output
            k1.append(foy[0][0])
            if(foy[0][0] <= 0.3):
                print("Benign")
            else:
                print("Malignant")

    def predict(self, inputs, outputs):                         #for confusion matrix
        outputs = np.array([outputs]).T
        k1 = []
        for j in range(len(inputs)):
            z = np.dot(self.weights0, (np.array([inputs[j]])).T) + self.biases0
            foz = sigmoid(z)                                    # output of hidden layer
            y = np.dot(self.weights1,foz)+self.biases1
            foy = sigmoid(y)                                    # final output
            k1.append(foy[0][0])
            k2 = outputs[j][0]

        (a, b, c, d) = (0, 0, 0, 0)

        for i in range(len(inputs)):
            if outputs[i][0] == 1:
                if k1[i] >= 0.3:
                    d = d + 1
                else:
                    c = c + 1
            else:
                if k1[i] <= 0.3:
                    a = a + 1
                else:
                    b = b + 1
        accuracy = (a + d) * 100 / (a + b + c + d)
        tpr = d * 100 / (c + d)
        fnr = c * 100 / (c + d)
        fpr = b * 100 / (a + b)
        tnr = a * 100 / (a + b)
        print("Accuracy =", accuracy, "%")
        print("   TPR   =", tpr, "%")
        print("   FNR   =", fnr, "%")
        print("   FPR   =", fpr, "%")
        print("   TNR   =", tnr, "%")

        return accuracy

# give inputs
#tr_inputs = X[:444]
#tr_outputs = Y[:444]
#test_inputs = X[444:]
# test_outputs = Y[444:]
tr_inputs, test_inputs, tr_outputs, test_outputs = train_test_split(X, Y, test_size=0.35)

#varied learning rate with 9 hidden nodes
ep1 = []
acc1 = []
n = NeuralNet(9)
for i in range(3, 13):
    n.train(tr_inputs, tr_outputs, i * 0.002, 2000, 0,9)
    print("learning rate =", i * 0.002)
    acc1.append(n.predict(test_inputs, test_outputs))
    ep1.append(i * 0.002)
acc1 = np.array(acc1)
ep1 = np.array(ep1)
fig, ax = plt.subplots()
ax.plot(ep1, acc1)
ax.set(xlabel='Learning rate', ylabel='Accuracy', title='Accuracy with varied learning rate (nodes=9)')
ax.grid()
plt.show()

#varied learning rate with 8 hidden nodes
ep1 = []
acc1 = []
n = NeuralNet(8)
for i in range(3, 13):
    n.train(tr_inputs, tr_outputs, i * 0.002, 2000, 0,8)
    print("learning rate =", i * 0.002)
    acc1.append(n.predict(test_inputs, test_outputs))
    ep1.append(i * 0.002)

acc1 = np.array(acc1)
ep1 = np.array(ep1)

fig, ax = plt.subplots()
ax.plot(ep1, acc1)
ax.set(xlabel='Learning rate', ylabel='Accuracy', title='Accuracy with varied learning rate (nodes=8)')
ax.grid()
plt.show()

#varied hidden nodes ith learning rate = 0.02
ep1 = []
acc1 = []
ex = []

for i in range(5, 12):
    n = NeuralNet(i)
    start = time.time()
    n.train(tr_inputs, tr_outputs, 0.02, 2000, 0,i)
    print('Hidden nodes=', i)
    acc1.append(n.predict(test_inputs, test_outputs))
    end = time.time()
    ex.append(end-start)
    ep1.append(i)

acc1 = np.array(acc1)
ep1 = np.array(ep1)
ex = np.array(ex)
fig, ax = plt.subplots()
ax.plot(ep1, acc1)
ax.set(xlabel='Hidden nodes', ylabel='Accuracy', title='Accuracy with varied Hidden Nodes (rate = 0.02)')
ax.grid()
plt.show()
fig1, ax1 = plt.subplots()
ax1.plot(ep1, ex)
ax1.set(xlabel='Hidden nodes', ylabel='Training and Testing Time', title='Execution Time with varied Hidden Nodes')
ax1.grid()
plt.show()


n = NeuralNet(9)
n.train(tr_inputs, tr_outputs, 0.02, 2000, 0,9)
n.predict(test_inputs, test_outputs)


def radialBasisFunc(K_cent):
    print("Radial Basis Function")
    km = KMeans(n_clusters=K_cent, max_iter=100)
    km.fit(tr_inputs)
    cent = km.cluster_centers_
    max = 0
    for i in range(K_cent):
        for j in range(K_cent):
            d = np.linalg.norm(cent[i] - cent[j])
            if d > max:
                max = d
    d = max
    sigma = d / math.sqrt(2 * K_cent)

    shape = tr_inputs.shape
    row = shape[0]
    column = K_cent
    G = np.empty((row, column), dtype=float)
    for i in range(row - 1):
        for j in range(column - 1):
            dist = np.linalg.norm(tr_inputs[i] - cent[j])
            G[i][j] = math.exp(-math.pow(dist, 2) / math.pow(2 * sigma, 2))
    GTG = np.dot(G.T, G)
    GTG_inv = np.linalg.inv(GTG)
    fac = np.dot(GTG_inv, G.T)
    W = np.dot(fac, tr_outputs)
    row = test_inputs.shape[0]
    column = K_cent
    G_test = np.empty((row, column), dtype=float)
    for i in range(row):
        for j in range(column):
            dist = np.linalg.norm(test_inputs[i] - cent[j])
            G_test[i][j] = math.exp(-math.pow(dist, 2) / math.pow(2 * sigma, 2))
    prediction = np.dot(G_test, W)
    c = 0
    for i in range(238):
        if test_outputs[i] == 0:
            if prediction[i] <= 0.3:
                c = c + 1
        else:
            if prediction[i] > 0.3:
                c = c + 1
    print("ACCURACY = " + str(c / 238))
hn = 9
radialBasisFunc(hn)

clf = svm.SVC(kernel='linear')                                  # Linear Kernel
clf.fit(tr_inputs, tr_outputs)                                  #Train the model using the training sets
y_pred = clf.predict(test_inputs)                               #Predict the response for test dataset
print(" SVM ")
print("Accuracy:",metrics.accuracy_score(test_outputs, y_pred))

classifier = DecisionTreeClassifier()
classifier.fit(tr_inputs, tr_outputs)
y_pred = classifier.predict(test_inputs)
print("Decision Tree")
print("Accuracy:",metrics.accuracy_score(test_outputs, y_pred))

#n.predict_1([[4,1,1,3,2,1,3,1,1],[8,10,10,8,7,10,9,7,1],[1,1,1,1,2,10,3,1,1]])
