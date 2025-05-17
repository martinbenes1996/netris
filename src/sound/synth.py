""""""

import matplotlib.pyplot as plt
import numpy as np
import simpleaudio as sa

frequency = 440  # tone frequency
fs = 44100  # samples per second
seconds = 1  # note duration

# variations in stable sound:
# 2000-4000 samples
# 44-88 ms
# variations in rythmic sound:
# 50-1000 samples
# 11.3

# time axis
t = np.arange(0, seconds, 1/fs)
# sine wave
x = np.sin(frequency * t * 2*np.pi)
print(x.shape)
exit()

# # quantize to 16bit
# x = x * (2**15-1) / np.max(np.abs(x))
# x = x.astype(np.int16)
# # visualize
# plt.plot(t, x)
# plt.show()

# frequency
y = np.fft.fft(x)
y_amp = np.abs(y)
# visualize
# yt = np.linspace(0, fs//2, y.shape[0]//2-1)
# plt.plot(yt, y_amp[:len(yt)])
# plt.yscale('log')
# plt.show()

# convert back to temporal
xx = np.fft.ifft(y)
print((x - xx).max())
# # visualize
# K = int(1/frequency*fs)
# plt.plot(t[:K], x[:K])
# plt.plot(t[:K], xx[:K])
# plt.plot(t[:K], xx[:K] - x[:K])
# plt.show()

# visualize

# # playback
# play_obj = sa.play_buffer(ft, 1, 2, fs)
# play_obj.wait_done()