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
            i=0
            for unit_property in unit_property_list:
                self.bias_property=self.bias_property+(unit_property[2],)
                if unit_property[2]=='un_biased':
                    self.i=np.empty((unit_property[0],unit_property[1]),dtype=object)
                elif unit_property[2]=='biased':
                    self.i=np.empty((unit_property[0],unit_property[1]),dtype=object)
                else:
                    print "Wrong Argument Input"
                    
                i=i+1
    
    def initialize_layer(self):
        for i in range(self.num_of_units):
            shape=self.i.shape
            for j in range(shape[0]):
                for k in range(shape[1]):
                    self.i[j][k]=node(self.layer_type,self.bias_property[i])
                    
            
    
    def create_connection(self,foreward_layer,connection_list)
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
        self.connection=tuple(connection_list)
        i=0                                                 #iteration over sub_units of current layer
        for unit_connection_tup in connection_list:
            # Error check for given tuple
            if len(unit_connection_tup) != foreward_layer.num_of_units :
                print ('Error.Give correct number of connection for each subunits of next layer')
            j=0                                             #iteration over sub_units of foreward_layer
            for unit_connection in unit_connection_tup:
                shape=foreward_layer.j.shape
                if unit_connection=='one_one':
                    theta_temp=np.random.rand(1,1)
                    #ErrorCheck
                    if foreward_layer.j.shape != self.i.shape:
                        print ("One to one corespondance is not possible")
                elif unit_connection=='one_to_all':
                    theta_temp=np.random.rand(shape[0],shape[1])
                unit_shape=self.j.shape
                for k in range(unit_shape[0]):
                    for l in range(unit_shape[1]):
                        self.i[k][l].theta=self.i[k][l].theta+(theta_temp,)
                        self.i[k][l].connection_type=self.i[k][l].connection_type+(unit_connection,)
                j=j+1
            i=i+1
            
                    
            