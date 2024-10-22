import matplotlib.pyplot as plt
import numpy as np

zebra = np.zeros((128,128))
for i in range(128):
    if i % 16 < 8:
        zebra[i, :] = 1
plt.imshow(zebra)
plt.savefig("ex2_f.pdf")
plt.show()

#exercitiul 3
#a
#  interval = 1/2000=0.0005
#b
#1 byte=8 biti
# 2000*4=8000 biti/sec
# =>1000 bytes/sec
# 1h=3600 sec
# 3600*1000=3600000 bytes