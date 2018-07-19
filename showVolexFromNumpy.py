import numpy as np

data = np.load('mlxFile/10_model.npy')
model_data = data[1].reshape(64,64,64) >0.5

print(model_data)