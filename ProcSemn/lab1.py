import matplotlib.pyplot as plt
import numpy as np

#Exercitiul 1
# a, b
# t = np.linspace(0, 0.5, int(0.5/0.003))
# x = np.cos(520 * np.pi * t + np.pi/3)
# y = np.cos(280 * np.pi * t - np.pi/3)
# z = np.cos(120 * np.pi * t + np.pi/3)
#
# fig, axs = plt.subplots(3)
# axs[0].plot(t, x)
# axs[1].plot(t, y)
# axs[2].plot(t, z)
#
# for ax in axs.flat:
#     ax.set_xlim([0,0.5])
# plt.savefig("ex1_a_b.pdf")
# plt.show()



# #c
# t = np.linspace(0, 1, 200)
# x = np.cos(520 * np.pi * t + np.pi/3)
# y = np.cos(280 * np.pi * t - np.pi/3)
# z = np.cos(120 * np.pi * t + np.pi/3)
#
# fig, axs = plt.subplots(3)
# axs[0].stem(t, x)
# axs[1].stem(t, y)
# axs[2].stem(t, z)
#
# for ax in axs.flat:
#     ax.set_xlim([0,1])
# plt.savefig("ex1_c.pdf")
# plt.show()


#Exercitiul 2

#a
# t = np.linspace(0, 4, 1600 )
# plt.stem(t, np.sin(2 * np.pi * 400 * t))
# plt.savefig("ex2_a.pdf")
# plt.show()

# #b
# t = np.linspace(0,3,2400)
# plt.stem(t, np.sin(2 * np.pi * 800 * t))
# plt.savefig("ex2_b.pdf")
# plt.show()

#c
# t = np.linspace(0,5/240,100)
# plt.plot(t,  (t%(1/240))/(1/240))
# plt.savefig("ex2_c.pdf")
# plt.show()

#d
# t = np.linspace(-1,1, 300)
# plt.plot(t, np.sign(np.sin(2 * np.pi * 300 * t)))
# plt.savefig("ex2_d.pdf")
# plt.show()

#e
# a = np.random.rand(128, 128)
# plt.imshow(a)
# plt.savefig("ex2_e.pdf")
# plt.show()

#f
# zebra = np.zeros((128,128))
# for i in range(128):
#     if i % 16 < 8:
#         zebra[i, :] = 1
# plt.imshow(zebra)
# plt.savefig("ex2_f.pdf")
# plt.show()

#exercitiul 3
#a
#  interval = 1/2000=0.0005
#b
#1 byte=8 biti
# 2000*4=8000 biti/sec
# =>1000 bytes/sec
# 1h=3600 sec
# 3600*1000=3600000 bytes

