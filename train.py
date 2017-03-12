from node import node
from layer import layer
import numpy as np
from network import network
import math
from itertools import izip

def create_input_layer_list(input_string):
    net_input=input_string.split(',')
    for i in range(len(net_input)):
        net_input[i]=float(net_input[i])
    list_of_input=[]
    for i in range(6):
        temp=tuple(net_input[64*i:64*(i+1)])
        list_of_input=list_of_input+[temp]
    return list_of_input
    
def create_output_layer_list(output_string):
    output_list=list(output_string)
    if output_list[0]=='p':
        init_file=int(output_list[1])
        final_file=int(output_list[2])
        final_rank=int(output_list[3])
        init_rank=final_rank-1
    elif output_list[0]=='R':
        init_rank=int(output_list[1])
        init_file=int(output_list[2])
        final_rank=int(output_list[3])
        final_file=int(output_list[4])
    elif output_list[0]=='N':
        init_rank=int(output_list[1])
        init_file=int(output_list[2])
        final_rank=int(output_list[3])
        final_file=int(output_list[4])
    elif output_list[0]=='B':
        init_rank=int(output_list[1])
        init_file=int(output_list[2])
        final_rank=int(output_list[3])
        final_file=int(output_list[4])
    elif output_list[0]=='Q':
        init_rank=int(output_list[1])
        init_file=int(output_list[2])
        final_rank=int(output_list[3])
        final_file=int(output_list[4])
    elif output_list[0]=='K':
        init_rank=int(output_list[1])
        init_file=int(output_list[2])
        final_rank=int(output_list[3])
        final_file=int(output_list[4])
    
    which=()
    where=()
    #print init_rank,init_file,final_rank,final_file
    for i in range(8):
        for j in range(8):
            if i==init_rank-1 and j==init_file-1:
                which=which+(1.0,)
            else:
                which=which+(0.0,)
            if i==final_rank-1 and j==final_file-1:
                where=where+(1.0,)
            else:
                where=where+(0.0,)
                
    list_of_output=[which,where]
    return list_of_output

#Variable and Definition
sigmoid=lambda x:1.0/(1+math.exp(-x))
sigmoid_gradient= lambda x:sigmoid(x)*(1-sigmoid(x))
batch_size=15
regularisation=10
descent_rate=0.25

#file_handling
input_handle=open('state1_input.txt','r')
output_handle=open('state1_output.txt','r')

#Initialization of layer 
a=layer('input',[(8,8,'un_biased'),(8,8,'un_biased'),(8,8,'un_biased'),(8,8,'un_biased'),(8,8,'un_biased'),(8,8,'un_biased')],batch_size)
b=layer('hidden',[(8,8,'biased'),(96,8,'un_biased')],batch_size)
c=layer('output',[(8,8,'un_biased'),(8,8,'un_biased')],batch_size)
a.initialize_layer()
b.initialize_layer()
c.initialize_layer()
a.create_connection(b,[('none','one_to_all'),('none','one_to_all'),('none','one_to_all'),('none','one_to_all'),('none','one_to_all'),('none','one_to_all')])
b.create_connection(c,[('one_to_all','one_to_all'),('one_to_all','one_to_all')])

#Initialization of Network
all_layer_list=[a,b,c]
net=network(all_layer_list,'stochastic',batch_size,regularisation,descent_rate)

epoch_count=0
for input,output in izip(input_handle,output_handle):
    #print 'HELLO KAY'
    #print output
    if output[0]=='p' or output[0]=='R' or output[0]=='N' or output[0]=='B' or output[0]=='Q' or output[0]=='K':
        #print output[0]
        list_of_input=create_input_layer_list(input)
        list_of_output=create_output_layer_list(output)
        net.in_batch_initialize_network()
        net.initialize_input_output_layer(list_of_input,list_of_output)
        net.network_foreward_propagate()
        net.network_back_propagate()
        c.cost_calculation()
        epoch_count+=1
    else:
        continue
        
    if epoch_count==batch_size:
        epoch_count=0
        print 'Cost Incurred: ',c.cost_incurred
        net.start_gradient_descent()
        net.batch_initialize_network()
