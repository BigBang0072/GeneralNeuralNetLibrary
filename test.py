from node import node
import numpy as np

a=node('hidden','biased')
a.z_val=1
print a.z_val
c=node('input','un_biased')


b=np.array((1,2))
print b
b[0]=a
b[1]=c

print b

