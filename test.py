from node import node
from layer import *
import numpy as np
from testC import testC


a=layer('input',[(1,1,'biased'),(2,1,'un_biased')])
b=layer('hidden',[(1,1,'biased'),(2,1,'un_biased')])
c=layer('output',[(2,1,'un_biased')])
a.initialize_layer()
b.initialize_layer()
c.initialize_layer()
a.create_connection(b,[('none','one_to_all'),('none','one_to_all')])
b.create_connection(c,[('one_to_all',),('one_to_all',)])
print "HAHAHAHAHAHAHAHHAH"
print b.sub_units[1][0][0].Gradient
print "HAHAHAHAHAHAHAHHAH"
A=a.sub_units
B=b.sub_units
C=c.sub_units
A[1][0][0].a_val=1
A[1][1][0].a_val=1
C[0][0][0].Y=1
C[0][1][0].Y=0
a.layer_foreward_propagate(b)
b.inlayer_foreward_propagate()
b.layer_foreward_propagate(c)
c.inlayer_foreward_propagate()
print "HAHAHAHAHAHAHAHHAH"
print B[1][0][0].Gradient
print "HAHAHAHAHAHAHAHHAH"
c.layer_back_propagate('none')
print "HAHAHAHAHAHAHAHHAH"
print B[1][0][0].Gradient
print "HAHAHAHAHAHAHAHHAH"
b.layer_back_propagate(c)
print "HAHAHAHAHAHAHAHHAH"
print B[1][0][0].Gradient
print "HAHAHAHAHAHAHAHHAH"
b.inlayer_back_propagate()
print "HAHAHAHAHAHAHAHHAH"
print B[1][0][0].Gradient
print "HAHAHAHAHAHAHAHHAH"
a.layer_back_propagate(b)
print "HAHAHAHAHAHAHAHHAH"
print B[1][0][0].Gradient
print "HAHAHAHAHAHAHAHHAH"
c.cost_calculation()
print "HAHAHAHAHAHAHAHHAH"
print B[1][0][0].Gradient
print "HAHAHAHAHAHAHAHHAH"

print A[0][0][0].Theta
print A[1][0][0].Theta
print A[1][1][0].Theta
print B[0][0][0].Theta
print B[1][0][0].Theta
print B[1][1][0].Theta

print A[0][0][0].Gradient
print A[1][0][0].Gradient
print A[1][1][0].Gradient
print B[0][0][0].Gradient
print B[1][0][0].Gradient
print B[1][1][0].Gradient

print "output_val:"
print C[0][0][0].a_val
print C[0][1][0].a_val
print "error_delta"
print C[0][0][0].error_delta
print C[0][1][0].error_delta


