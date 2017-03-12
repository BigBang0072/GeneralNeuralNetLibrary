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
net=network(all_layer_list,'stochastic',1,1,1)

#for debugging
net.all_layer_tup[0].sub_units[1][0][0].a_val=2
net.all_layer_tup[0].sub_units[1][1][0].a_val=3
net.all_layer_tup[2].sub_units[0][0][0].Y=1
net.all_layer_tup[2].sub_units[0][1][0].Y=0
A=a.sub_units
B=b.sub_units
C=c.sub_units

print "Theta"
print A[0][0][0].Theta
print A[1][0][0].Theta
print A[1][1][0].Theta
print B[0][0][0].Theta
print B[1][0][0].Theta
print B[1][1][0].Theta

print "Error_delta"
print B[0][0][0].error_delta
print B[1][0][0].error_delta
print B[1][1][0].error_delta
print C[0][0][0].error_delta
print C[0][1][0].error_delta
print "Gradient"
print A[0][0][0].Gradient
print A[1][0][0].Gradient
print A[1][1][0].Gradient
print B[0][0][0].Gradient
print B[1][0][0].Gradient
print B[1][1][0].Gradient

print "Last a val"
print C[0][0][0].a_val
print C[0][1][0].a_val
print "cost"
print c.cost_incurred

print "Foreward_propagate"
net.network_foreward_propagate()
net.network_back_propagate()

print "cost"
print c.cost_incurred
print ""
print ""
print ""
print"BACK PROPAGATED"

print "Last a val"
print C[0][0][0].a_val
print C[0][1][0].a_val

print "Error_delta"
print B[0][0][0].error_delta
print B[1][0][0].error_delta
print B[1][1][0].error_delta
print C[0][0][0].error_delta
print C[0][1][0].error_delta

print "Gradient"
print A[0][0][0].Gradient
print A[1][0][0].Gradient
print A[1][1][0].Gradient
print B[0][0][0].Gradient
print B[1][0][0].Gradient
print B[1][1][0].Gradient

print "Theta"
print A[0][0][0].Theta
print A[1][0][0].Theta
print A[1][1][0].Theta
print B[0][0][0].Theta
print B[1][0][0].Theta
print B[1][1][0].Theta

print "cost"
print c.cost_incurred
print ""
print ""
print ""

print "#############APPLYING GRADIENT DESCENT#############"


net.start_gradient_descent()
print "Theta"
print A[0][0][0].Theta
print A[1][0][0].Theta
print A[1][1][0].Theta
print B[0][0][0].Theta
print B[1][0][0].Theta
print B[1][1][0].Theta



print "Gradient"
print A[0][0][0].Gradient
print A[1][0][0].Gradient
print A[1][1][0].Gradient
print B[0][0][0].Gradient
print B[1][0][0].Gradient
print B[1][1][0].Gradient
print c.cost_incurred

print ""
print ""
print ""
print "Reinitializing the network for the new example in the batch"

print "cost"
print c.cost_incurred
net.initialize_network()
print "Last a val"
print C[0][0][0].a_val
print C[0][1][0].a_val

print "Error_delta"
print B[0][0][0].error_delta
print B[1][0][0].error_delta
print B[1][1][0].error_delta
print C[0][0][0].error_delta
print C[0][1][0].error_delta

print "Gradient"
print A[0][0][0].Gradient
print A[1][0][0].Gradient
print A[1][1][0].Gradient
print B[0][0][0].Gradient
print B[1][0][0].Gradient
print B[1][1][0].Gradient

print "Theta"
print A[0][0][0].Theta
print A[1][0][0].Theta
print A[1][1][0].Theta
print B[0][0][0].Theta
print B[1][0][0].Theta
print B[1][1][0].Theta







