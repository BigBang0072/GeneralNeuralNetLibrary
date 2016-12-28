import numpy as np
from node import node
import math

class layer():
    def __init__(self,layer_type,unit_property_list):
        ''' Argument 1: It gives the type of layer it is out of (input,hidden,output)
            Argument 2: It is a list of dimension of subunits in a layer in form of tuples for each subunit like (number_of rows,number_of_column,'biased or un_biased')'''
            
        self.num_of_units=len(unit_property_list)
        self.layer_type=layer_type
        self.bias_property=()
        self.sub_units=()
        self.connections=()
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
        self.connections=tuple(connection_list)
        
        #iteration over sub_units of current layer
        for i,unit_connection_tup in enumerate(connection_list):
            #print unit_connection_tup
            # Error check for given tuple
            if len(unit_connection_tup) != foreward_layer.num_of_units :
                print ('Error.Give correct number of connection for each subunits of next layer')
            
            #iteration over sub_units of foreward_layer
            for j,unit_connection in enumerate(unit_connection_tup):
                shape=foreward_layer.sub_units[j].shape
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
                unit_shape=self.sub_units[i].shape
                #print unit_shape
                for k in range(unit_shape[0]):
                    for l in range(unit_shape[1]):
                        #print 'Hi K'
                        theta_temp=theta_temp*np.random.rand()
                        self.sub_units[i][k][l].Theta=self.sub_units[i][k][l].Theta+(theta_temp,)
                        self.sub_units[i][k][l].Gradient=self.sub_units[i][k][l].Gradient+(gradient_temp,)
                        self.sub_units[i][k][l].connection_type=self.sub_units[i][k][l].connection_type+(unit_connection,)
                
            
            
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
                                    
            
                    
            