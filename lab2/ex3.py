import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.io import wavfile
import os
def play_and_save(signal, fs, filename):
    sd.play(signal, fs)
    sd.wait()
    wavfile.write(filename, fs, signal.astype(np.float32))
folder = "pdfs"
os.makedirs(folder, exist_ok=True)

# a
fs_a = 44100
t_a = np.linspace(0, 4, 1600)
signal_a = np.sin(2 * np.pi * 400 * t_a)

plt.stem(t_a, signal_a)
plt.savefig(os.path.join(folder, "ex3_a.pdf"))
plt.show()

play_and_save(signal_a, fs_a, "ex3_a.wav")

# b
fs_b =44100
t_b = np.linspace(0, 3, 800)
signal_b = np.sin(2 * np.pi * 800 * t_b)

plt.stem(t_b, signal_b)
plt.savefig(os.path.join(folder,"ex3_b.pdf"))
plt.show()

play_and_save(signal_b, fs_b, "ex3_b.wav")

# c
fs_c = 44100
t_c = np.linspace(0, 5/240, 100)
signal_c = (t_c % (1/240)) / (1/240)

plt.plot(t_c, signal_c)
plt.savefig(os.path.join(folder,"ex3_c.pdf"))
plt.show()

play_and_save(signal_c, fs_c, "ex3_c.wav")

# d
fs_d = 44100
t_d = np.linspace(-1, 1, 300)
signal_d = np.sign(np.sin(2 * np.pi * 300 * t_d))

plt.plot(t_d, signal_d)
plt.savefig(os.path.join(folder,"ex3_d.pdf"))
plt.show()

play_and_save(signal_d, fs_d, "ex3_d.wav")



fs_loaded, signal_loaded = wavfile.read("ex3_a.wav")
print(f"Frecvența de eșantionare: {fs_loaded}")
print(f"Semnal încărcat: {signal_loaded[:10]}...")
