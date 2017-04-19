from node import node
from layer import layer
import numpy as np
from network import network
import math

sigmoid=lambda x:1.0/(1+math.exp(-x))
sigmoid_gradient= lambda x:sigmoid(x)*(1-sigmoid(x))


a=layer('input',[(1,1,'biased'),(100,1,'un_biased')],1)
b=layer('hidden',[(1,1,'biased'),(40,1,'un_biased')],1)
c=layer('output',[(10,1,'un_biased')],1)
a.initialize_layer()
b.initialize_layer()
c.initialize_layer()
a.create_connection(b,[('none','one_to_all'),('none','one_to_all')])
b.create_connection(c,[('one_to_all',),('one_to_all',)])

all_layer_list=[a,b,c]
net=network(all_layer_list,'stochastic',1,0,.025)
net.initialize_input_output_layer([(),tuple(range(100))],[(0,1,0,0,0,0,0,0,0,0)])

for i in range(100):
    net.network_foreward_propagate()
    net.network_back_propagate()
    net.start_gradient_descent()
    
    print "out1:",c.sub_units[0][0][0].a_val
    print "out2:",c.sub_units[0][1][0].a_val
    print "out3:",c.sub_units[0][2][0].a_val
    print "out4:",c.sub_units[0][3][0].a_val
    print "out5:",c.sub_units[0][4][0].a_val
    print "out6:",c.sub_units[0][5][0].a_val
    print "out7:",c.sub_units[0][6][0].a_val
    print "out8:",c.sub_units[0][7][0].a_val
    print "out9:",c.sub_units[0][8][0].a_val
    print "out10:",c.sub_units[0][9][0].a_val
    #c.cost_calculation()
    #print "cost=",c.cost_incurred
    #c.cost_incurred=0
    print "################################################"
    
