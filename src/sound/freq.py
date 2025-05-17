""""""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def fft(x, window_size=1024, step_size=128):
	#
	num_samples = len(x)
	num_steps = int(np.ceil(num_samples / step_size))
	num_windows = int(np.ceil(num_samples / window_size))
	# iterate over windows
	y = np.zeros((num_steps, window_size//2), dtype='complex')
	for k in range(num_steps):
		k_start = k*step_size
		y_k = np.fft.fft(x[k_start:k_start+window_size])
		y[k, :len(y_k)//2] = y_k[:len(y_k)//2]
	return y

if __name__ == '__main__':
	# generate sound
	Fs = 44100  # Hz
	t = np.arange(0, 10, 1/Fs)
	x = np.sin(440 * t * 2*np.pi)
	# convert to frequency
	y = np.abs(fft(x, step_size=128, window_size=1024))
	# to longer
	df = (
		pd.DataFrame(y)
		.reset_index(drop=False)
		.melt(id_vars='index')
		.rename({
			'index': 'step',
			'variable': 'freq',
			'value': 'val'
		}, axis=1)
	)
	# convert scales
	df['freq'] = df['freq'].astype('float64')
	df['freq'] = df['freq'] / (1024//2) * (Fs//2)
	# convert values
	df['val'] = df['val'].astype('float64')
	# to wider
	df = df.pivot(index='step', columns='freq', values='val')
	#
	fig, ax = plt.subplots()
	sns.heatmap(df, xticklabels=100, ax=ax)
	# ax.set_xticklabels(df.columns.map(lambda v: f'{v:.02f}'))
	plt.show()
