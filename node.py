import numpy as np
import math
class node:
    #ATTRIBUTES
    
    # Single Valued Attributes
    z_val=0
    a_val=0
    error_delta=0                                      # Gradient upto the end of that node(backward)
    # Array of Attributes
    Theta=np.zeros((1,1),Float)
    gradient=np.zeros((1,1),Float)
    
    #METHODS
    
    #Initialization
    def __init__(self,which_node,node_val):
    
        ''' ARGUMENT 1: which_node : take the STRING with position given possibility=('input','hidden','output')
            ARGUMENT 2: node_val : Only active when the node type is input or output(in those case give the corresponding X and Y value of those nodes in that layer) '''
        
        possibility=('input','hidden','output')
        if which_node in possibility:
            self.node_position=which_node
            if which_node=='input':
                self.X=node_val
            elif which_node=='output'
                
        else:
            print("Error.Position not well defined")
            
    #Foreward Propagation (in same layer)
    def foreward_propagate(self,rectification function_name,rectification_function)
    
        '''ARGUMENT 1: Give rectification name if its inbuilt else give 'NEW' as argument.
           ARGUMENT 2: Give new rectification function handle "using lambda" or if inbuilt give 'NONE'. '''
           
           sigmoid=lambda x: 1/(1-math.exp(x)
           available function=(sigmoid)
           if rectification_function_name in available_function:
                self.a_val=rectification_function_name(self.a_val)
           else:
                self.a_val=rectification_function(self.z_val)
    
    #Finding Gradient
    def fi(self)
           