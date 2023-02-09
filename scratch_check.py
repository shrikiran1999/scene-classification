import numpy as np

# a = np.array([0,1,2])
# b = np.array([3,4,5])
# l = [a,b]
# c = np.concatenate(l)
# print(c)
# print(l)

arr = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15]])

# arr = np.array([np.array(i) for i in range(16)])

# print(arr)
# print(np.hstack((arr[0], arr[1])))
# print(np.reshape(arr, (4, 2,2)))
# # new_arr = np.reshape(arr, (4, 4))
# arr2 = np.reshape(arr, (4, 2, 2))
# new_arr = np.reshape(arr, 16, order='F')
# # new_new_arr = np.reshape(new_arr, (4, 2, 2), order='A')
# print(new_arr)
# # print(np.hstack((arr2[0], arr2[1])))

# arr = np.array([1,-2,0])
# print(np.maximum(arr, np.zeros_like(arr)))
# l = ['k', 1,2]
# a,b,c = l
# print(type(a))
# print(b)
# print(c)

# arr = np.zeros((7,7,512))
# print(arr.T.shape)
# print(np.transpose(arr, (2,0,1)).flatten().shape)

# a = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
# b = a[::-1]
# print(b)
# b = np.array([[1, 2], [1,2]])

# print(np.argmin(np.linalg.norm((a-b),axis=1)))
# print(np.pad(a, ((3,1), (1,1)), 'constant', constant_values=0))

# arr = np.arange(0, 16, 1)
# print(arr)
# new_arr = np.reshape(arr, (4,4) )
# print(new_arr)
# new_patches = []
# # print(np.reshape(new_arr, 16, order='F'))
# for i in range(0, new_arr.shape[0], 2):
#     for j in range(0, new_arr.shape[1], 2):
#         sub_patch_row1 = np.hstack((new_arr[i][j], new_arr[i][j+1]))
#         sub_patch_row2 = np.hstack((new_arr[i+1][j], new_arr[i+1][j+1]))
#         new_patches.append(np.vstack((sub_patch_row1, sub_patch_row2)))

# print(new_patches)
# l = 1
# patch_idxs = np.array([i for i in range(4**(l+1))])
# patch_idxs = np.reshape(patch_idxs, (int((4**(l+1))**0.5), int((4**(l+1))**0.5)))
# print(patch_idxs)

A = np.array([[0.1,0.4,0.5], [0.1,0.4,0.5]])
B = np.array([[0.2,0.3,0.5],[0.8,0.1,0.1]])

C = np.minimum(A,B)
sim = np.sum(C, axis=1)
print(sim)
print(C)


ValueError --------------------------------------------------------------------------- ValueError                                
Traceback (most recent call last) <ipython-input-1-8ff2fe0c588d> in <module>      
54     print('Test Case passed. Good job!')      
55  ---> 56 unittest_build_recognition_system()      
57 ### END HIDDEN TESTS  <ipython-input-1-8ff2fe0c588d> in unittest_build_recognition_system()      
14         count += 1      
15  ---> 16     trained_features = trained_system['features']      
17       18     train_labels = trained_system['labels']  /usr/local/lib/python3.6/dist-packages/numpy/lib/npyio.py in __getitem__(self, key)     
253                 return format.read_array(bytes,     
254                                          allow_pickle=self.allow_pickle, --> 
255                                          pickle_kwargs=self.pickle_kwargs)     
256             else:     
257                 return self.zip.read(key)  /usr/local/lib/python3.6/dist-packages/numpy/lib/format.py in read_array(fp, allow_pickle, pickle_kwargs)     
725         # The array contained Python objects. We need to unpickle the data.     
726         if not allow_pickle: --> 727             raise ValueError("Object arrays cannot be loaded when "     
728                              "allow_pickle=False")     
729         if pickle_kwargs is None:  ValueError: Object arrays cannot be loaded when allow_pickle=False