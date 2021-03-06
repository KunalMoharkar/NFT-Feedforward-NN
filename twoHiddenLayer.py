import numpy as np
import data as Data

#initail nodes parameters
num_nodes_input = 100
num_nodes_hidden_layer_1 = 80
num_nodes_hidden_layer_2 = 30
num_nodes_output = 3

#get data from data.py file
Dataset = Data.Dataset
Target = Data.Target
Testset = Data.Testset
Expectedoutput = Data.Expectedoutput

#all the values of learning rate
alpha = [0.01,0.05,0.1,0.2,0.4,0.8]

#activation function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#function to train 
def train(W,Wb,V,Vb,Dataset,Target,alpha):
    num_nodes_input = len(W)
    num_nodes_hidden_layer_1 = len(Wb)
    num_nodes_hidden_layer_2 = len(Vb)
    num_nodes_output = len(Ub)

    epoch = 0
    condition = True
    finalop = [[]]*15
    Z1 = [0]*num_nodes_hidden_layer_1
    Z2 = [0]*num_nodes_hidden_layer_2
    Y = [0]*num_nodes_output
    delta_output = [0]*num_nodes_output
    delta_hidden_layer_1 = [0]*num_nodes_hidden_layer_1
    delta_hidden_layer_2 = [0]*num_nodes_hidden_layer_2


    while condition is True:                   #continue till error less than 0.01 or epaoch < 10000
        error = 0
        epoch += 1
        for num in range(15):                  # for each input vector

            X=Dataset[num]
            T=Target[num]
                                                        #forward propogation
            for j in range(num_nodes_hidden_layer_1):
                Zinj = 0
                for i in range(num_nodes_input):
                    Zinj += X[i]*W[i][j]

                Zinj = Wb[j] + Zinj
                Z1[j] = sigmoid(Zinj)
            
            for j in range(num_nodes_hidden_layer_2):
                Zinj = 0
                for i in range(num_nodes_hidden_layer_1):
                    Zinj += Z1[i]*V[i][j]

                Zinj = Vb[j] + Zinj
                Z2[j] = sigmoid(Zinj)

            for j in range(num_nodes_output):
                Yinj = 0
                for i in range(num_nodes_hidden_layer_2):
                    Yinj += Z2[i]*U[i][j]

                Yinj = Ub[j] + Yinj
                Y[j] = sigmoid(Yinj)
            

            finalop[num] = Y.copy()

            for i in range(3):                            #sum square error
                error+=(finalop[num][i] - T[i])**2

            for k in range(num_nodes_output):                       #backword Propogate
                delta_output[k] = (T[k]-Y[k])*(Y[k])*(1-Y[k])

            for j in range(num_nodes_hidden_layer_2):
                deltaj = 0
                for k in range(num_nodes_output):
                    deltaj += delta_output[k]*U[j][k]
                delta_hidden_layer_2[j] = deltaj
            
            for j in range(num_nodes_hidden_layer_2):
                delta_hidden_layer_2[j] = delta_hidden_layer_2[j]*(Z2[j])*(1-Z2[j])

            for j in range(num_nodes_hidden_layer_1):
                deltaj = 0
                for k in range(num_nodes_hidden_layer_2):
                    deltaj += delta_hidden_layer_2[k]*V[j][k]
                delta_hidden_layer_1[j] = deltaj
            
            for j in range(num_nodes_hidden_layer_1):
                delta_hidden_layer_1[j] = delta_hidden_layer_1[j]*(Z1[j])*(1-Z1[j])

            for i in range(num_nodes_input):                                 #update weights
                for j in range(num_nodes_hidden_layer_1):   
                    W[i][j]+=alpha*delta_hidden_layer_1[j]*X[i]    
            
            for i in range(num_nodes_hidden_layer_1):
                Wb[i]+=alpha*delta_hidden_layer_1[i]
            
            for j in range(num_nodes_hidden_layer_1):
                for k in range(num_nodes_hidden_layer_2):
                    V[j][k]+=alpha*delta_hidden_layer_2[k]*Z1[j]

            for i in range(num_nodes_hidden_layer_2):
                Vb[i]+=alpha*delta_hidden_layer_2[i]

            for j in range(num_nodes_hidden_layer_2):
                for k in range(num_nodes_output):
                    U[j][k]+=alpha*delta_output[k]*Z2[j]

            for i in range(num_nodes_output):
                Ub[i]+=alpha*delta_output[i]
            

        if error<0.01 or epoch>10000:                           #breaking condition
            condition=False    
    print(f"Epoch count is :{epoch}")

#function to test
def test(W,Wb,V,Vb,Testset):

    Z1 = [0]*num_nodes_hidden_layer_1
    Z2 = [0]*num_nodes_hidden_layer_2
    Y = [0]*num_nodes_output
    Yconverted = [0]*num_nodes_output
    num = 0

    for X in Testset:
        for j in range(num_nodes_hidden_layer_1):
            Zinj = 0
            for i in range(num_nodes_input):
                Zinj += X[i]*W[i][j]
            Zinj = Wb[j] + Zinj
            Z1[j] = sigmoid(Zinj)
            
        for j in range(num_nodes_hidden_layer_2):
            Zinj = 0
            for i in range(num_nodes_hidden_layer_1):
                Zinj += Z1[i]*V[i][j]
            Zinj = Vb[j] + Zinj
            Z2[j] = sigmoid(Zinj)

        for j in range(num_nodes_output):
            Yinj = 0
            for i in range(num_nodes_hidden_layer_2):
                Yinj += Z2[i]*U[i][j]
            Yinj = Ub[j] + Yinj
            Y[j] = sigmoid(Yinj)
            
            if Y[j]<0.5:
                Yconverted[j] = 0
            else:
                Yconverted[j] = 1
                                                                                #direct to appropriate output
        if Yconverted[0]==0 and Yconverted[1]==0 and Yconverted[2]==1:
            result = "K"
        elif Yconverted[0]==0 and Yconverted[1]==1 and Yconverted[2]==0:
            result = "D"
        elif Yconverted[0]==1 and Yconverted[1]==0 and Yconverted[2]==0:
            result = "M"
        else :
            result = "Cannot Identify"

        print(f"Decimal output:{Y}   and output is: {result}  Output should be: {Expectedoutput[num]}")
        num+=1

        
#for all the learning rates
for i in range(6):
    #initialize weight and bias with random values
    W = np.random.rand(num_nodes_input,num_nodes_hidden_layer_1)
    W = np.multiply(W,0.01) #multiply entire matrix with 0.01 to reduce initial weights
    Wb = np.random.rand(num_nodes_hidden_layer_1)
    Wb = np.multiply(Wb,0.01)
    V = np.random.rand(num_nodes_hidden_layer_1,num_nodes_hidden_layer_2)
    V = np.multiply(V,0.01)
    Vb = np.random.rand(num_nodes_hidden_layer_2)
    Vb = np.multiply(Vb,0.01)
    U = np.random.rand(num_nodes_hidden_layer_2,num_nodes_output)
    U = np.multiply(U,0.01)
    Ub = np.random.rand(num_nodes_output)
    Ub = np.multiply(Ub,0.01)
    print("\n#############################################################################\n")
    print(f"the learning rate is {alpha[i]}\n")
    train(W,Wb,V,Vb,Dataset,Target,alpha[i])
    test(W,Wb,V,Vb,Testset)

    