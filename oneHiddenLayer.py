import numpy as np  #import numpy
import data as Data #import feature vectors

#initail nodes parameters
num_nodes_input = 100
num_nodes_hidden = 80
num_nodes_output = 3
#get all the feature vectors from data.py
Dataset = Data.Dataset
Target = Data.Target
Testset = Data.Testset
Expectedoutput = Data.Expectedoutput

#all the values of learning rate
alpha = [0.01,0.05,0.1,0.2,0.4,0.8]

#sigmoid activation function
def sigmoid(x):        
    return 1/(1+np.exp(-x))

#function to train Network
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
    
    while condition is True:                 #continue while error is < 0.01 0r epoch < 10000
        error = 0 
        epoch += 1
        for num in range(15):                #for each input image vector

            X=Dataset[num]                  
            T=Target[num]

                                                     #forward Propogation
            for j in range(num_nodes_hidden):        #caluculate Z
                Zinj = 0
                for i in range(num_nodes_input):
                    Zinj += X[i]*W[i][j]

                Zinj = Wb[j] + Zinj
                Z[j] = sigmoid(Zinj)
            
            for j in range(num_nodes_output):        #calculate Y
                Yinj = 0
                for i in range(num_nodes_hidden):
                    Yinj += Z[i]*V[i][j]

                Yinj = Vb[j] + Yinj
                Y[j] = sigmoid(Yinj)

            finalop[num] = Y.copy()                  #stores output for each of 15 image vectors

            for i in range(3):                        #sum square error
                error+=(finalop[num][i] - T[i])**2
            
            for k in range(num_nodes_output):                        # back Propogation
                delta_output[k] = (T[k]-Y[k])*(Y[k])*(1-Y[k])        #errors for output nodes

            for j in range(num_nodes_hidden):                       
                deltaj = 0
                for k in range(num_nodes_output):
                    deltaj += delta_output[k]*V[j][k]
                delta_hidden[j] = deltaj          
            
            for j in range(num_nodes_hidden):
                delta_hidden[j] = delta_hidden[j]*(Z[j])*(1-Z[j])      #errors for hidden nodes
 
            for i in range(num_nodes_input):                           #udate weights and Bias
                for j in range(num_nodes_hidden):
                    W[i][j]+=alpha*delta_hidden[j]*X[i]                
            
            for i in range(num_nodes_hidden):
                Wb[i]+=alpha*delta_hidden[i]
            
            for j in range(num_nodes_hidden):
                for k in range(num_nodes_output):
                    V[j][k]+=alpha*delta_output[k]*Z[j]

            for i in range(num_nodes_output):
                Vb[i]+=alpha*delta_output[i]

        if error<0.01 or epoch>10000:                                    #stopping condition
            condition=False        
    print(f"Epoch is {epoch}")


#function to test the network
def test(W,Wb,V,Vb,Testset):

    Z = [0]*num_nodes_hidden
    Y = [0]*num_nodes_output
    Yconverted = [0]*num_nodes_output
    num=0
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
            
            if Y[j]<0.5:                             #approximate values based on output
                Yconverted[j] = 0
            else:
                Yconverted[j] = 1


        if Yconverted[0]==0 and Yconverted[1]==0 and Yconverted[2]==1:        #output cases
            result = "K"
        elif Yconverted[0]==0 and Yconverted[1]==1 and Yconverted[2]==0:
            result = "D"
        elif Yconverted[0]==1 and Yconverted[1]==0 and Yconverted[2]==0:
            result = "M"
        else :
            result = "Cannot Identify"

        print(f"Decimal output:{Y}   and output is: {result}  Output should be: {Expectedoutput[num]}")
        num+=1

        

    
#test for all the learning rates
for i in range(6):
    #initialize with random values
    W = np.random.rand(num_nodes_input,num_nodes_hidden)
    W = np.multiply(W,0.01)
    Wb = np.random.rand(num_nodes_hidden)
    Wb = np.multiply(Wb,0.01)
    V = np.random.rand(num_nodes_hidden,num_nodes_output)
    V = np.multiply(V,0.01)
    Vb = np.random.rand(num_nodes_output)
    Vb = np.multiply(Vb,0.01)

    print("\n#############################################################################\n")
    print(f"the learning rate is {alpha[i]}\n")
    train(W,Wb,V,Vb,Dataset,Target,alpha[i])
    test(W,Wb,V,Vb,Testset)