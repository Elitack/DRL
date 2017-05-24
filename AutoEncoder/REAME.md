#AutoEncoder

##Function:
Provide the AutoEncoder funcition for the given dataset.

##Usage

###import
from AutoEncoder.autoencoder import AutoEncoder

###Initialize
test = AutoEncoder(config=config)

####config
*case write should be considered*

#####hiddenSize
+ usage:the num units for different layer.
+ type:list(int)
+ the size of the first input layer also **should be included**
+ **must need**
+ the number of the lists are not compulsion.

#####trainData
+ usage: give the input data
+ type:list(float)
+ **must need**
+ the number and the dimension of the input data are free to run.

#####reg
+ usage: for the regularization
+ type:float
+ optional need, if not given in config ,the default is 0.001

#####activation
+ usage:for the activation function
+ type:tensorflow function
+ optional need, if not given in config, the default is tf.nn.elu

#####regularizer
+ usage:for the regularization
+ type:tensorflow function
+ optional need, if not given in config, the default is tf.contrib.layers.l2_regularizer

#####initializer
+ usage:initialize the weights
+ type:tensorflwo function
+ optional need, if not given in config, the default is tf.contrib.layers.variance_scaling_initializer()

#####learning_rate
+ usage:for the learning
+ type:float
+ optional need, if not given in config, the default is 0.01

#####epoch
+ usage:for the learning
+ type:int
+ optional need, if not given in config, the default is 5

#####batchSize
+ usage:for the learning
+ type:int
+ optional need, if not given in config, the default is 150


###Train and Learn
test.learn()

###Get the parameter
+ usage:test.getParameter()
+ type: weights_dict, biases_dict

weights_dict:{'weights1': xxx, 'weights2':xxx, ......}
biases_dict:{'biases1': xxx, 'biases2':xxx, ......}

###Instance Usage:
```python
if __name__ == '__main__':
    config = {}
    config['hiddenSize'] = [1, 10, 7] #set 3 layers
    config['trainData'] = np.array([float(i) for i in range(1000)]).reshape(-1, config['hiddenSize'][0]).tolist() #set the training data:0,1,2,3,4....999
    testSample = AutoEncoder(config=config)
    testSample.learn()
    print (testSample.getParameter())
```

 

