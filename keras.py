
import numpy as np 

myarray = np.fromfile('/home/aneri/Downloads/mobilenet_1_0_224_tf.h5', dtype=float)
print(myarray)
print(len(myarray))

np.savetxt("weights.txt", myarray, newline=" ")
