import numpy as np
from node import node
import math

class layer():
    def __init__(self,layer_type,unit_property_list):
        ''' Argument 1: It gives the type of layer it is out of (input,hidden,output)
            Argument 2: It is a list of dimension of subunits in a layer in form of tuples for each subunit like (number_of rows,number_of_column,'biased or un_biased')'''
            
        self.num_of_units=len(unit_property_list)
        self.layer_type=layer_type
        if layer_type=='output':
            self.cost_incurred=0
        self.bias_property=()
        self.sub_units=()
        self.connections=()
        sigmoid=lambda x:1.0/(1+math.exp(-x))
        sigmoid_gradient= lambda x:sigmoid(x)*(1-sigmoid(x))
        self.rectification_function_tup=(sigmoid,)              # Add as many rectification function as necessary in the tuple.
        self.rectification_gradient_tup=(sigmoid_gradient,)
        #print "HI"
        for i,unit_property in enumerate(unit_property_list):
            self.bias_property=self.bias_property+(unit_property[2],)
            if unit_property[2]=='un_biased':
                self.sub_units=self.sub_units+(np.empty((unit_property[0],unit_property[1]),dtype=object),)
            elif unit_property[2]=='biased':
                self.sub_units=self.sub_units+(np.empty((unit_property[0],unit_property[1]),dtype=object),)
            else:
                print "Wrong Argument Input"
                
    
    def initialize_layer(self):
        ''' Look for the better method for initialization of the layer'''
        for i in range(self.num_of_units):
            shape=self.sub_units[i].shape
            for j in range(shape[0]):
                for k in range(shape[1]):
                    self.sub_units[i][j][k]=node(self.layer_type,self.bias_property[i])
                    #print "hi"
                      
    def create_connection(self,foreward_layer,connection_list):
        ''' Argument 1: Give the foreward layer object to which we want to connection_list
            Argument 2: its a list of tuple of connection from each subunits in this layer to all the subunits in foreward layer
            Possible connection now are ('one_one','one_to_all','none') {subunit to subunit connection}
            'one_one' : creates one to one mapping from current subunit to the specified subunit in next layer
            'one_to_all : creates mapping where this all the nodes of this subunit are mapped to all the nodes of next layer subunit.
            Later 'convolution' will be added. '''
        
        # Error Checking in argument
        if len(connection_list) != self.num_of_units:
            print ('Error.Give correct number of connection for each subunits of this layer')
        
        # Starting to connect
        self.connections=tuple(connection_list)  #for protection from assignment by mistake by user
        
        #iteration over sub_units of current layer
        for i,unit_connection_tup in enumerate(connection_list):
            #print unit_connection_tup
            # Error check for given tuple
            if len(unit_connection_tup) != foreward_layer.num_of_units :
                print ('Error.Give correct number of connection for each subunits of next layer')
            
            #iteration over sub_units of foreward_layer
            for j,unit_connection in enumerate(unit_connection_tup):
                shape=foreward_layer.sub_units[j].shape
                unit_shape=self.sub_units[i].shape
                #print unit_shape
                #Initializing the theta and gradient matrices seperately for each element(inefficient), as shape will be same for all the elements in that subunit's theta for that particular subunit in foreward_layer. But it is created by reference and all the nodes's theta will be essentially same. Think for better way.
                for k in range(unit_shape[0]):
                    for l in range(unit_shape[1]):
                        if unit_connection=='one_one':
                            theta_temp=np.random.rand(1,1)
                            gradient_temp=np.random.rand(1,1)
                            #print theta_temp
                            #ErrorCheck
                            if foreward_layer.sub_units[j].shape != self.sub_units[i].shape:
                                print ("One to one corespondance is not possible")
                        elif unit_connection=='one_to_all':
                            theta_temp=np.random.rand(shape[0],shape[1])
                            gradient_temp=np.random.rand(shape[0],shape[1])
                            #print theta_temp
                        elif unit_connection=='none':
                            theta_temp=np.random.rand(0,0)
                            gradient_temp=np.random.rand(0,0)
                        #print 'Hi K'
                        self.sub_units[i][k][l].Theta=self.sub_units[i][k][l].Theta+(theta_temp,)
                        self.sub_units[i][k][l].Gradient=self.sub_units[i][k][l].Gradient+(gradient_temp,)
                        self.sub_units[i][k][l].connection_type=self.sub_units[i][k][l].connection_type+(unit_connection,)#??Why are we keeping this information with the node also as it is available with teh layer in layer's connection tuple.
                
                      
    # Foreward_propagation of layer to next layer
    def layer_foreward_propagate(self,foreward_layer):
        for i,sub_unit_current in enumerate(self.sub_units):
            unit_connection_tup=self.connections[i]
            for j,sub_unit_foreward in enumerate(foreward_layer.sub_units):
                unit_connection=unit_connection_tup[j]
                shape_unit_current=self.sub_units[i].shape
                shape_unit_foreward=foreward_layer.sub_units[j].shape
                if unit_connection =='one_one':
                    for k in range(shape_unit_current[0]):
                        for l in range(shape_unit_current[1]):
                            foreward_layer.sub_units[j][k][l].z_val=foreward_layer.sub_units[j][k][l].z_val+(self.sub_units[i][k][l].a_val*self.sub_units[i][k][l].Theta[j].item(0))
                elif unit_connection=='one_to_all':
                    for k in range(shape_unit_current[0]):
                        for l in range(shape_unit_current[1]):
                            for m in range(shape_unit_foreward[0]):
                                for n in range(shape_unit_foreward[1]):
                                    foreward_layer.sub_units[j][m][n].z_val=foreward_layer.sub_units[j][m][n].z_val+(self.sub_units[i][k][l].a_val*self.sub_units[i][k][l].Theta[j].item(m,n))
                                    
    
    # ONLY FOR HIDDEN layer and OUTPUT layer. Rectifying the z_val to convert to a_val.
    def inlayer_foreward_propagate(self):
        if self.layer_type=='hidden' or self.layer_type=='output':
            for i,sub_unit in enumerate(self.sub_units):
                shape=sub_unit.shape
                if self.bias_property[i]=='un_biased':
                    for j in range(shape[0]):
                        for k in range(shape[1]):
                            sub_unit[j][k].a_val=self.rectification_function_tup[0](sub_unit[j][k].z_val)
                else:
                    print ""
        else:
            print ("inlayer activation not valid for input layer")
            
    # For our good old cost function J of classification problem. For other cost function write your own error_delta. on the output nodes.       
    def layer_back_propagate(self,foreward_layer):
        ''' Argument 1: is the foreward_layer from which we want error to be back-propagated, to find gradient of theta in this current layer.
                        In case of output layer instead of foreward layer just give none as input.
            BEWARE: We have to use it wisely by sequentially back-propagating from the last layer to hte first layer.
                    For our good old cost function J of classification problem. For other cost function write your own error_delta. on the output nodes.'''
        if self.layer_type=='output':
            for i,sub_unit in enumerate(self.sub_units):
                shape=sub_unit.shape
                for j in range(shape[0]):
                    for k in range(shape[1]):
                        # Gradient of J(cost function: our standard).For other cost function write your own error_delta. on the output nodes. 
                        sub_unit[j][k].error_delta=sub_unit[j][k].a_val-sub_unit[j][k].Y
        elif self.layer_type=='hidden':
            for i,sub_unit_current in enumerate(self.sub_units):
                unit_connection_tup=self.connections[i]
                for j,sub_unit_foreward in enumerate(foreward_layer.sub_units):
                    unit_connection=unit_connection_tup[j]
                    shape_unit_current=sub_unit_current.shape
                    shape_unit_foreward=sub_unit_foreward.shape
                    if unit_connection=='one_one':
                        for k in range(shape_unit_current[0]):
                            for l in range(shape_unit_current[1]):
                                self.sub_units[i][k][l].Gradient[j][0][0]=foreward_layer.sub_units[j][k][l].error_delta*self.sub_units[i][k][l].a_val
                                self.sub_units[i][k][l].error_delta=self.sub_units[i][k][l].error_delta+(foreward_layer.sub_units[j][k][l].error_delta*self.sub_units[i][k][l].Theta[j][0][0].item(0))
                    elif unit_connection=='one_to_all':
                        for k in range(shape_unit_current[0]):
                            for l in range(shape_unit_current[1]):
                                for m in range(shape_unit_foreward[0]):
                                    for n in range(shape_unit_foreward[1]):
                                        #print "error_delta: ",foreward_layer.sub_units[j][m][n].error_delta
                                        #print 'a_val: ',self.sub_units[i][k][l].a_val
                                        #print 'i,k,l: ',i,k,l
                                        #print 'j,m,n: ',j,m,n
                                        #print 'totuk: ',self.sub_units[1][0][0].Gradient
                                        self.sub_units[i][k][l].Gradient[j][m][n]=foreward_layer.sub_units[j][m][n].error_delta*self.sub_units[i][k][l].a_val
                                        #print 'self.sub_units[i][k][l].Gradient[j][m][n]: ',self.sub_units[i][k][l].Gradient[j][m][n]
                                        #print 'totuk: ',self.sub_units[1][0][0].Gradient
                                        print "ERROR CHECK"
                                        print "Initial Error delta, Foreward_error_delta, Theta :::",self.sub_units[i][k][l].error_delta,foreward_layer.sub_units[j][m][n].error_delta,self.sub_units[i][k][l].Theta[j].item(m,n)
                                        self.sub_units[i][k][l].error_delta=self.sub_units[i][k][l].error_delta+foreward_layer.sub_units[j][m][n].error_delta*self.sub_units[i][k][l].Theta[j].item(m,n)
                        
        elif self.layer_type=='input':
            for i,sub_unit_current in enumerate(self.sub_units):
                unit_connection_tup=self.connections[i]
                for j,sub_unit_foreward in enumerate(foreward_layer.sub_units):
                    unit_connection=unit_connection_tup[j]
                    shape_unit_current=sub_unit_current.shape
                    shape_unit_foreward=sub_unit_foreward.shape
                    if unit_connection=='one_one':
                        for k in range(shape_unit_current[0]):
                            for l in range(shape_unit_current[1]):
                                self.sub_units[i][k][l].Gradient[j][0][0]=foreward_layer.sub_units[j][k][l].error_delta*self.sub_units[i][k][l].a_val
                                #self.sub_units[i][k][l].error_delta=self.sub_units[i][k][l].error_delta+(foreward_layer.sub_units[j][k][l].error_delta*self.sub_units[i][k][l].Theta[j][0][0].item(0))
                    elif unit_connection=='one_to_all':
                        for k in range(shape_unit_current[0]):
                            for l in range(shape_unit_current[1]):
                                for m in range(shape_unit_foreward[0]):
                                    for n in range(shape_unit_foreward[1]):
                                        self.sub_units[i][k][l].Gradient[j][m][n]=foreward_layer.sub_units[j][m][n].error_delta*self.sub_units[i][k][l].a_val
                                        #self.sub_units[i][k][l].error_delta=self.sub_units[i][k][l].error_delta+foreward_layer.sub_units[j][m][n].error_delta*self.sub_units[i][k][l].Theta[j].item(m,n)
          
    def inlayer_back_propagate(self):
        ''' To get the error delta with sigmoid gradient applied which was left in layer_back_propagate'''
        if self.layer_type =='hidden':
            for i,sub_unit in enumerate(self.sub_units):
                shape=sub_unit.shape
                for j in range(shape[0]): 
                    for k in range(shape[1]):
                        sub_unit[j][k].error_delta=self.rectification_gradient_tup[0](sub_unit[j][k].a_val)*sub_unit[j][k].error_delta
                    
    def cost_calculation(self):
        '''Only valid for output layer'''
        if self.layer_type=='output':
            for i,sub_unit in enumerate(self.sub_units):
                shape=sub_unit.shape
                for j in range(shape[0]):
                    for k in range(shape[1]):
                        #print sub_unit[j][k].a_val
                        #print ((1-sub_unit[j][k].Y)*math.log(1-sub_unit[j][k].a_val)+(sub_unit[j][k].Y)*math.log(sub_unit[j][k].a_val))
                        self.cost_incurred=self.cost_incurred-((1-sub_unit[j][k].Y)*math.log(1-sub_unit[j][k].a_val)+(sub_unit[j][k].Y)*math.log(sub_unit[j][k].a_val))
        else:
            print "This function is only defined for the output layer. Tin ton tin"
     

     
def gradient_checking(all_layer_list):
    ''' To check if the model is consistent by calculating gradient numerically and then comparing it to gradient calculated trough back propagation'''
    epsilon=math.pow(10,-4)     #as directed in exercise by Ng sahab.
    for i,layer in enumerate(all_layer_list):
        for j,sub_unit in enumerate(layer.sub_units):
            unit_shape=sub_unit.shape
            for k in range(unit_shape[0]):
                for l in range(unit_shape[1]):
                    for m,theta in enumerate(sub_unit[k][l].Theta):
                        theta_shape=theta.shape
                        for n in range(theta_shape[0]):
                            for o in range(theta_shape[1]):
                                #print all_layer_list
                                temp_theta_holder= theta[n][o]
                                theta[n][o]=temp_theta_holder+epsilon
                                cost_epsilon_plus=give_cost(all_layer_list)
                                #print all_layer_list
                                all_layer_list.reverse()
                                theta[n][o]=temp_theta_holder-epsilon
                                cost_epsilon_minus=give_cost(all_layer_list)
                                all_layer_list.reverse()
                                theta[n][o]=temp_theta_holder
                                cost=give_cost(all_layer_list)
                                all_layer_list.reverse()
                                numerical_gradient=(cost_epsilon_plus-cost_epsilon_minus)/(2*epsilon)
                                actual_gradient=sub_unit[k][l].Gradient[m][n][o]
                                if math.fabs(numerical_gradient-actual_gradient)<0.0009:
                                    flag='PASSED'
                                else:
                                    flag='FAILED'
                                
                                print ("Actual:",actual_gradient," Numerical:",numerical_gradient," Status:",flag)
                                
def give_cost(all_layer_list):
    for i,layer in enumerate(all_layer_list):
        if layer.layer_type=='input':
            layer.layer_foreward_propagate(all_layer_list[i+1])
        elif layer.layer_type=='hidden':
            layer.inlayer_foreward_propagate()
            layer.layer_foreward_propagate(all_layer_list[i+1])
        else:
            layer.inlayer_foreward_propagate()
            
    
    
    all_layer_list.reverse()
    all_layer_list[0].cost_calculation()
    for i,layer in enumerate(all_layer_list):
        if layer.layer_type=='input':
            layer.layer_back_propagate(all_layer_list[i-1])
        elif layer.layer_type=='hidden':
            layer.layer_back_propagate(all_layer_list[i-1])
            layer.inlayer_back_propagate()
        else:
            layer.layer_back_propagate('none')
    
    return all_layer_list[0].cost_incurred
    
            