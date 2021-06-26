import numpy as np
import scipy.io
from copy import copy
# data generation
N = 64
Total_set = 600000
#sparsity = 14
l = 1 # frame length

# method 1 : variational sparsity (Beronulli sampling)

# Data X(modulated in BPSK) can be generated from the lines below, but I used Matlab instead
'''
agumented_BPSK = [1, -1, 0]
X = np.zeros((Total_set,N))
i = 0
while i < Total_set:
	X[i] = np.random.choice(agumented_BPSK, N, p=[0.15, 0.15, 0.7])
	if np.sum(np.abs(X[i])) == 0:
		pass
	else:
		i = i + 1
'''

# import X.mat generated from matlab
X_ = scipy.io.loadmat('x_32_12_32_8.mat')
X = X_['x_32_12_32_8']
X = X.reshape((Total_set, N))

#X_pilot = copy(X)
#X_pilot[np.nonzero(X)] = 1
#X_frame = np.concatenate((X_pilot, X,X,X,X,X,X,X), axis=1)
X_frame = copy(X)
X_frame = X_frame.reshape((Total_set, l, N))

train_test_ratio = 9/10
train_valid_ratio = 8/9
n_split_test = int(X.shape[0]*train_test_ratio)
n_split_valid = int(n_split_test*train_valid_ratio)

X_train = X[:n_split_valid]
X_train_frame = X_frame[:n_split_valid]
Y_train = copy(X_train)
Y_train[np.nonzero(Y_train)] = 1

X_valid = X[n_split_valid:n_split_test]
X_valid_frame = X_frame[n_split_valid:n_split_test]
Y_valid = copy(X_valid)
Y_valid[np.nonzero(X_valid)] = 1

X_test = X[n_split_test:]
X_test_frame = X_frame[n_split_test:]
Y_test = copy(X_test)
Y_test[np.nonzero(X_test)] = 1

data = {
	'train' : {'x' : X_train_frame, 'y' : Y_train},
	'valid' : {'x' : X_valid_frame, 'y' : Y_valid},
	'test' : {'x' : X_test_frame, 'y' : Y_test}
}

print('Training dataset:\n-------------------')
print('x:', data['train']['x'].shape)
print('y:', data['train']['y'].shape)

print('Validation dataset:\n-------------------')
print('x:', data['valid']['x'].shape)
print('y:', data['valid']['y'].shape)

print('Test dataset:\n-------------------')
print('x:', data['test']['x'].shape)
print('y:', data['test']['y'].shape)
print(data['train']['x'][0])
print(data['train']['y'][0])
print(data['test']['x'][0])
print(data['test']['y'][0])

np.save('dataset_32_12_32_8.npy', data)





# The lines below were not used in the paper in which dataset from Beronulli sampling was used
# method2: fixed sparsity and frame length L ( p pilot symbols and d data symbols)  p+d = L
'''
X = np.zeros((Total_set,N))
for j in range(Total_set):
	idx = np.random.choice(N, size=sparsity, replace=False)
	X[j, idx] = np.random.choice([1, -1], sparsity, p=[0.5, 0.5])

X_pilot = copy(X)
X_pilot[np.nonzero(X)] = 1
#X_frame = np.concatenate((X_pilot, X,X,X,X,X,X,X), axis=1)
X_frame = copy(X)
X_frame = X_frame.reshape((Total_set, l, N))

#train_test_ratio = 9/10
train_test_ratio = 1/7
#train_valid_ratio = 8/9
train_valid_ratio = 6/7
n_split_test = int(X.shape[0]*train_test_ratio)
n_split_valid = int(n_split_test*train_valid_ratio)

X_train = X[:n_split_valid]
X_train_frame = X_frame[:n_split_valid]
Y_train = copy(X_train)
Y_train[np.nonzero(Y_train)] = 1

X_valid = X[n_split_valid:n_split_test]
X_valid_frame = X_frame[n_split_valid:n_split_test]
Y_valid = copy(X_valid)
Y_valid[np.nonzero(X_valid)] = 1

X_test = X[n_split_test:]
X_test_frame = X_frame[n_split_test:]
Y_test = copy(X_test)
Y_test[np.nonzero(X_test)] = 1

data = {
	'train' : {'x' : X_train_frame, 'y' : Y_train},
	'valid' : {'x' : X_valid_frame, 'y' : Y_valid},
	'test' : {'x' : X_test_frame, 'y' : Y_test}
}

print('Training dataset:\n-------------------')
print('x:', data['train']['x'].shape)
print('y:', data['train']['y'].shape)

print('Validation dataset:\n-------------------')
print('x:', data['valid']['x'].shape)
print('y:', data['valid']['y'].shape)

print('Test dataset:\n-------------------')
print('x:', data['test']['x'].shape)
print('y:', data['test']['y'].shape)

print(data['train']['x'][0])
print(data['train']['y'][0])
print(data['test']['x'][0])
print(data['test']['y'][0])
np.save('dataset_real_fixed_sparsity_14_N_64_p0_d1.npy', data)
'''
