from node import node
from layer import layer
import numpy as np

class network():
    def __init__(self,all_layer_list,backpropagation_method,batch_size,regulariastion_val,descent_rate):
        ''' Argument 1: (list)Give the list of all layer sequentially in a list eg(input_layer,hidden_layer1,... ... ,hidden_layer_last,output_layer)
            Argument 2: (String)This argument will let the network know if we are backpropagating batch wise or stochastically
            Argument 3: (integer) In case of batch gradient descent give the size of batch or in case of stochastic descent give the data_input size'''
        self.all_layer_tup=tuple(all_layer_list)
        self.backpropagation_method=backpropagation_method
        self.batch_size=float(batch_size)
        self.lambda_val=float(regulariastion_val)
        self.alpha_rate=float(descent_rate)
        
    
    def initialize_input_output_layer(self,file_read_handle):
        #both X val and Y val for new elemnt of output layer
        for lines in fhad:
            initialize()
        
        
    #There is no need to create a new network and set the Thetas from previous network. We will just initialize this present network 
    # to the state that just by initializing the input_layer with new batch we can resume our work.
    def initialize_network(self):
        ''' To set the network to initial state to start a fresh epoch cycle by reseting all the variable '''
        for layer in self.all_layer_tup:
            for i,sub_unit in enumerate(layer.sub_units):
                shape_current_unit=sub_unit.shape
                for j in range(shape_current_unit[0]):
                    for k in range(shape_current_unit[1]):
                        if layer.layer_type=='hidden':
                            sub_unit[j][k].error_delta=0
                            if layer.bias_property[i]=='un_biased':
                                sub_unit[j][k].a_val=0
                                sub_unit[j][k].z_val=0
                        elif layer.layer_type=='output':
                            sub_unit[j][k].error_delta=0
                            sub_unit[j][k].a_val=0
                            sub_unit[j][k].z_val=0
                                                    
           
    def network_foreward_propagate(self):
        for i,layer in enumerate(self.all_layer_tup):
            if layer.layer_type=='input':
                layer.layer_foreward_propagate(self.all_layer_tup[i+1])
            elif layer.layer_type=='hidden':
                layer.inlayer_foreward_propagate()
                layer.layer_foreward_propagate(self.all_layer_tup[i+1])
            elif layer.layer_type=='output':
                layer.inlayer_foreward_propagate()
                
        #(IMPORTANT) Calculate the cost function and DONT divide by the batch size.(or while implementing) in final script (to full cost as  I have divided the m factor from regularisation part also.)
                
    def network_back_propagate(self):
        for i,reversed_layer in enumerate(reversed(self.all_layer_tup)):
            if reversed_layer.layer_type=='output':
                reversed_layer.layer_back_propagate('none')
            elif reversed_layer.layer_type=='hidden':
                reversed_layer.layer_back_propagate(self.all_layer_tup[i-1])
                reversed_layer.inlayer_back_propagate()
            elif reversed_layer.layer_type=='input':
                reversed_layer.layer_back_propagate(self.all_layer_tup[i-1])
    
    def start_gradient_descent(self):
        num_units=len(self.all_layer_tup)
        output_layer=self.all_layer_tup[num_units-1]
        for i,layer in enumerate(self.all_layer_tup):
            if layer.layer_type != 'output':
                for unit_index,sub_unit in enumerate(layer.sub_units):
                    shape_current_unit=sub_unit.shape
                    for j in range(shape_current_unit[0]):
                        for k in range(shape_current_unit[1]):
                            for theta_index,theta in enumerate(sub_unit[j][k].Theta)
                                shape_theta=theta.shape
                                if (shape_theta[0]+shape_theta[1])>0:
                                    for l in range(shape_theta[0]):
                                        for m in range(shape_theta[1]):
                                            if layer.bias_property[unit_index] != 'biased':
                                                output_layer.cost_incurred=output_layer.cost_incurred+((self.lambda_val/(2*self.batch_size))*(sub_unit[j][k].Theta[theta_index].item(l,m)**2))
                                                sub_unit[j][k].Gradient[theta_index][l][k]=sub_unit[j][k].Gradient[theta_index].item(l,m)+((self.lambda_val/self.batch_size)*sub_unit[j][k].Theta[theta_index].item(l,m))
                                            sub_unit[j][k].Theta[theta_index][l][m]=sub_unit[j][k].Theta[theta_index].item(l,m)-((self.alpha_rate)*sub_units[j][k].Gradient[theta_index].item(l,m))
    
    def calculate_cost(self):