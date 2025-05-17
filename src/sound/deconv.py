
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage, signal, fft, fftpack
import seaborn as sns

def show(x):
	sns.heatmap(x, cmap='icefire', square=True)
	plt.show()

def convolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    print(star_fft.shape, psf_fft.shape)
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft*psf_fft)))

def deconvolve(star, psf):
    star_fft = fftpack.fftshift(fftpack.fftn(star))
    psf_fft = fftpack.fftshift(fftpack.fftn(psf))
    return fftpack.fftshift(fftpack.ifftn(fftpack.ifftshift(star_fft/psf_fft)))

# load image
x = np.array(Image.open('/Users/martin/Datasets/lena.png').convert('L'))
x = x.astype('float64')
x_mu, x_sd = x.mean(), x.std()
# x = (x - x_mu) / x_sd

# convolve
KB = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])
xf = convolve(x, KB)

y_xf = deconvolve(xf, KB)
sns.heatmap(y_xf, cmap='icefire', square=True)
plt.show()
exit()

# convolve in frequency domain
y_xf = fft.fft2(xf)
y_KB = fft.fft2(KB, s=x.shape)

# deconvolve in frequency domain
x_xf = np.abs(fft.ifft2(y_xf * y_KB))
x_xf = x_xf / np.std(x_xf) * x_sd
x_xf = np.clip(x_xf, 0, 255)
print(x_xf.mean())
# show
fig, ax = plt.subplots(1, 2)
sns.histplot(x.flatten(), ax=ax[0])
sns.histplot(np.abs(x_xf).flatten(), ax=ax[1])
plt.show()
exit()

# deconvolve
xf_l = ndimage.laplace(xf)
KB_l = ndimage.laplace(KB)
print(xf_l.shape, KB_l.shape)

# # deconvolve
# xx = signal.wiener(xf, KB.shape)
# xx = xx * x_sd + x_mu
# xx = (xx - xx.min()) / (xx.max() - xx.min()) * 255
# show(xx)
# plt.imshow(xx, cmap='gray')
# plt.show()


# # perform
# KBi = np.linalg.pinv(KB, hermitian=True)
# print(KB @ KBi)  # ~= np.eye(3)

# sns.heatmap(xf, cmap='icefire', square=True)#, norm=mpl.colors.LogNorm())
# # plt.imshow(xf, cmap='gray')
# plt.show()