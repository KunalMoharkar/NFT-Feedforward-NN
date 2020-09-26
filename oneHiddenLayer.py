import numpy as np
import data as Data

#initail nodes parameters
num_nodes_input = 100
num_nodes_hidden = 80
num_nodes_output = 3

#initialize weight and bias with random values
W = np.random.rand(num_nodes_input,num_nodes_hidden)
W = np.multiply(W,0.01)
Wb = np.random.rand(num_nodes_hidden)
Wb = np.multiply(Wb,0.01)
V = np.random.rand(num_nodes_hidden,num_nodes_output)
V = np.multiply(V,0.01)
Vb = np.random.rand(num_nodes_output)
Vb = np.multiply(Vb,0.01)

Dataset = Data.Dataset
Target = Data.Target
alpha = 0.3
a = [1]

print(W)

def sigmoid(x):  #sigmoid activation function
    return 1/(1+np.exp(-x))

def train(W,Wb,V,Vb,Dataset,Target,alpha):
    

    condition = True
    epoch = 0
    num_nodes_input = len(W)
    num_nodes_hidden = len(Wb)
    num_nodes_output = len(Vb)
    finalop = [[]]*15
    Z = [0]*num_nodes_hidden
    Y = [0]*num_nodes_output
    delta_output = [0]*num_nodes_output
    delta_hidden = [0]*num_nodes_hidden
    

    while condition is True:
        error = 0 
        epoch += 1
        for num in range(15):

          
            X=Dataset[num]
            T=Target[num]

            for j in range(num_nodes_hidden):
                Zinj = 0
                for i in range(num_nodes_input):
                    Zinj += X[i]*W[i][j]

                Zinj = Wb[j] + Zinj
                Z[j] = sigmoid(Zinj)
            
            for j in range(num_nodes_output):
                Yinj = 0
                for i in range(num_nodes_hidden):
                    Yinj += Z[i]*V[i][j]

                Yinj = Vb[j] + Yinj
                Y[j] = sigmoid(Yinj)

            finalop[num] = Y.copy()

            for i in range(3):
                error+=(finalop[num][i] - T[i])**2
            
            for k in range(num_nodes_output):
                delta_output[k] = (T[k]-Y[k])*(Y[k])*(1-Y[k])

            for j in range(num_nodes_hidden):
                deltaj = 0
                for k in range(num_nodes_output):
                    deltaj += delta_output[k]*V[j][k]
                delta_hidden[j] = deltaj
            
            for j in range(num_nodes_hidden):
                delta_hidden[j] = delta_hidden[j]*(Z[j])*(1-Z[j])

            for i in range(num_nodes_input):
                for j in range(num_nodes_hidden):
                    W[i][j]+=alpha*delta_hidden[j]*X[i]    
            
            for i in range(num_nodes_hidden):
                Wb[i]+=alpha*delta_hidden[i]
            
            for j in range(num_nodes_hidden):
                for k in range(num_nodes_output):
                    V[j][k]+=alpha*delta_output[k]*Z[j]

            for i in range(num_nodes_output):
                Vb[i]+=alpha*delta_output[i]

        if error<0.01:
            condition=False        
        print(error)
    print(f"Epoch is {epoch}")



def test(W,Wb,V,Vb,Testset):

    Z = [0]*num_nodes_hidden
    Y = [0]*num_nodes_output
    for X in Testset:
        for j in range(num_nodes_hidden):
            Zinj = 0
            for i in range(num_nodes_input):
                Zinj += X[i]*W[i][j]
                Zinj = Wb[j] + Zinj
            Z[j] = sigmoid(Zinj)
            
        for j in range(num_nodes_output):
            Yinj = 0
            for i in range(num_nodes_hidden):
                Yinj += Z[i]*V[i][j]
                Yinj = Vb[j] + Yinj
            Y[j] = sigmoid(Yinj)
    


         

train(W,Wb,V,Vb,Dataset,Target,alpha)
