import numpy as np

a =np.array([[1,2,3],[2,3,4],[3,4,5],[0,0,0],[7,8,9]])
# print(np.argwhere(a[:,0] ==0))


print(a[:,0].all()== a[:,1].all()==a[:,2].all())
a =np.delete(a,[],axis=0)

print(a)