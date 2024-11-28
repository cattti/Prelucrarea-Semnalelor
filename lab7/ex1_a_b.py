import numpy as np
import matplotlib.pyplot as plt
# X = misc.face(gray=True)
# plt.imshow(X, cmap=plt.cm.gray)
# plt.show()


#transformata fourier
# Y = np.fft.fft2(X)
# freq_db = 20*np.log10(abs(Y))

# plt.imshow(freq_db)
# plt.colorbar()
# plt.show()


#rotire la 45 grade
# rotate_angle = 45
# X45 = ndimage.rotate(X, rotate_angle)
# plt.imshow(X45, cmap=plt.cm.gray)
# plt.show()
#
# Y45 = np.fft.fft2(X45)
# plt.imshow(20*np.log10(abs(Y45)))
# plt.colorbar()
# plt.show()


#Momentan pe axe sunt afișate numărul bin-urilor
#  Pentru a obține frecvențele asociate folosiți `fftfreq`
# freq_x = np.fft.fftfreq(X.shape[1])
# freq_y = np.fft.fftfreq(X.shape[0])
#
# plt.stem(freq_x, freq_db[:][0])
# plt.show()





# fara frecvente inalte
# freq_cutoff = 120
#
# Y_cutoff = Y.copy()
# Y_cutoff[freq_db > freq_cutoff] = 0
# X_cutoff = np.fft.ifft2(Y_cutoff)
# X_cutoff = np.real(X_cutoff)    # avoid rounding erros in the complex domain,
#                                 # in practice use irfft2
# plt.imshow(X_cutoff, cmap=plt.cm.gray)
# plt.show()



# adaugam zgomot in limita pixel noise
# pixel_noise = 200
#
# noise = np.random.randint(-pixel_noise, high=pixel_noise+1, size=X.shape)
# X_noisy = X + noise
# plt.imshow(X, cmap=plt.cm.gray)
# plt.title('Original')
# plt.show()
# plt.imshow(X_noisy, cmap=plt.cm.gray)
# plt.title('Noisy')
# plt.show()


n1 = 72
n2 = 72
x = np.linspace(0, 1, n1)
y = np.linspace(0, 1, n2)


def f1(x, y):
    return np.sin(2 * np.pi * x + 3 * np.pi * y)


def f2(x, y):
    return np.sin(4 * np.pi * x) + np.cos(6 * np.pi * y)


X, Y = np.meshgrid(x, y)

f_1 = f1(X, Y)
f_2 = f2(X, Y)

spectrum_f1 = np.real(np.fft.fftshift(np.fft.ifft2(f_1)))
spectrum_f2 = np.real(np.fft.fftshift(np.fft.ifft2(f_2)))

fig, axs = plt.subplots(2, 2)
axs[0][0].imshow(f_1)
axs[1][0].imshow(f_2)
axs[0][1].imshow(spectrum_f1)
axs[1][1].imshow(spectrum_f2)
plt.savefig("ex1_a_b.pdf")
plt.show()







