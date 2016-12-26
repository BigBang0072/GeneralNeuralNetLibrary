import numpy as np
import math
class node:
    #ATTRIBUTES
    connection_type=()      
    # Array of Attributes
    Theta=()                # It will be tuple of arrays (where arrays will hold theta theta for a particular subunits)
    gradient=()
    
    #METHODS
    
    #Initialization
    def __init__(self,which_node,biased_flag):
    
        ''' ARGUMENT 1: which_node : take the STRING with position given possibility=('input','hidden','output') 
            ARGUMENT 2: biased flag gives sting input if the node is 'biased' or 'un_biased' '''
        possibility=('input','hidden','output')
        if which_node in possibility:
            self.node_position=which_node
            if which_node=='input':
                # I think we should keep it as arrays for multiple input examples handling at same time for the  batch 
                if biased_flag=='un_biased':
                    self.X=0
                elif biased_flag=='biased':
                    self.X=1
            elif which_node=='output':
                self.z_val=0
                self.h_val=0
                self.Y=0
                self.error_delta=0                  # Gradient upto the end of that node(backward)
            else:
                if biased_flag=='un_biased':
                    self.z_val=0
                    self.a_val=0
                    self.error_delta=0              # Gradient upto the end of that node(backward)
                elif biased_flag=='biased':
                    self.a_val=1
                    self.error_delta=0              # Gradient upto the end of that node(backward)
        else:
            print("Error.Position not well defined")
            
    #Foreward Propagation (in same layer)
    def foreward_propagate(self,rectification_function_name,rectification_function):
    
        '''ARGUMENT 1: Give rectification name if its inbuilt else give 'NEW' as argument.
           ARGUMENT 2: Give new rectification function handle "using lambda" or if inbuilt give 'NONE'. '''
           
        sigmoid=lambda x: 1/(1-math.exp(x)
        available_function=('sigmoid')
        if rectification_function_name in available_function:
            self.a_val=rectification_function_name(self.a_val)
        else:
            self.a_val=rectification_function(self.z_val)
    
    #Finding Gradient
    #def find_gradient(self,just_foreward_layer)
    
        ''' Finding Gradient for all the theta which is going out of this node 
            ARGUMENT 1: We take the '''
           