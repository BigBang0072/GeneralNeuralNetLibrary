from node import node
from layer import layer
import numpy as np
from network import network
import math

sigmoid=lambda x:1.0/(1+math.exp(-x))
sigmoid_gradient= lambda x:sigmoid(x)*(1-sigmoid(x))


a=layer('input',[(1,1,'biased'),(2,1,'un_biased')],1)
b=layer('hidden',[(1,1,'biased'),(2,1,'un_biased')],1)
c=layer('output',[(2,1,'un_biased')],1)
a.initialize_layer()
b.initialize_layer()
c.initialize_layer()
a.create_connection(b,[('none','one_to_all'),('none','one_to_all')])
b.create_connection(c,[('one_to_all',),('one_to_all',)])

all_layer_list=[a,b,c]
net=network(all_layer_list,'stochastic',1,0,.025)
net.initialize_input_output_layer([(),tuple(range(2))],[(0,1)])

