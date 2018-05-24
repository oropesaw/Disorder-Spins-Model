import numpy as np
import matplotlib.pyplot as plt
import sys

from matplotlib import rc
from astropy.convolution import convolve, Gaussian1DKernel

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
rc("text", usetex=True)

plt.style.use("seaborn-darkgrid")

def heat_capacity(names, smooth=False, sigma=5):
	
	for i, name in enumerate(names):
		data = np.loadtxt(name, dtype=float)
		num = float(name.split('.')[0])
		contrt = num/32.0
		_x = data[:,1];		_y = data[:,7]

		if smooth:
			gauss_kernel = Gaussian1DKernel(sigma)
			plt.plot(_x, convolve(_y, gauss_kernel), lw='2' ,label=r'$x=${:.3f}'.format(contrt))
		else :
			plt.plot(_x, _y, lw='2' ,label=r'$x=${:.3f}'.format(contrt))

	plt.xlabel(r'$T$', fontsize=25)
	plt.ylabel(r'$C/N$', fontsize=25)
	plt.xlim(xmin=2.0, xmax=5.0)
	plt.xticks(size=25)
	plt.yticks(size=25)
	#plt.legend(fontsize=18)
	plt.tight_layout()
	plt.show()


def susceptibility(names, smooth=False, sigma=5):
	
	for i, name in enumerate(names):
		data = np.loadtxt(name, dtype=float)
		num = float(name.split('.')[0])
		contrt = num/32.0
		_x = data[:,1];		_y = data[:,8]

		if smooth:
			gauss_kernel = Gaussian1DKernel(sigma)
			plt.plot(_x, convolve(_y, gauss_kernel), lw='2' ,label=r'$x=${:.3f}'.format(contrt))
		else :
			plt.plot(_x, _y, lw='2' ,label=r'$x=${:.3f}'.format(contrt))

	plt.xlabel(r'$T$', fontsize=25)
	plt.ylabel(r'$\chi/N$', fontsize=25)
	plt.xlim(xmin=2.0, xmax=5.0)
	plt.xticks(size=25)
	plt.yticks(size=25)
	#plt.legend(fontsize=18)
	plt.tight_layout()
	plt.show()


def heat_capacity_per_T(names, smooth=False, sigma=5):
	
	for i, name in enumerate(names):
		data = np.loadtxt(name, dtype=float)
		num = float(name.split('.')[0])
		contrt = num/32.0
		_x = data[:,1];		_y = data[:,7]

		if smooth:
			gauss_kernel = Gaussian1DKernel(sigma)
			plt.plot(_x, convolve(_y/_x, gauss_kernel), lw='2' ,label=r'$x=${:.3f}'.format(contrt))
		else :
			plt.plot(_x, _y/_x, lw='2' ,label=r'$x=${:.3f}'.format(contrt))

	plt.xlabel(r'$T$', fontsize=25)
	plt.ylabel(r'$C/NT$', fontsize=25)
	plt.xlim(xmin=2.0, xmax=5.0)
	plt.xticks(size=25)
	plt.yticks(size=25)
	#plt.legend(fontsize=18)
	plt.tight_layout()
	plt.show()


def temperature():
	_x = []
	_y = []
	for i in range(1,33):
		data = np.loadtxt(str(i) + '.out', dtype=float)
		contrt = i/32.0
		_x.append(contrt)
		_y.append(data[:,1][np.argmax(data[:,7])]/2.269)

	# Ajuste lineal de los puntos
	poly = np.poly1d(np.polyfit(_x, _y, 1))
	plt.plot(_x, _y, 's')
	#plt.plot(_x, poly(_x), label='ajust')
	#plt.legend(fontsize=18)
	plt.xlabel(r'$x$', fontsize=25)
	plt.ylabel(r'$T_{c}/T^{(2D)}_{C}$', fontsize=25)
	plt.xticks(size=25)
	plt.yticks(size=25)
	plt.tight_layout()
	plt.show()


def energy(names, smooth=False, sigma=5):
	
	for i, name in enumerate(names):
		data = np.loadtxt(name, dtype=float)
		num = float(name.split('.')[0])
		contrt = num/32.0
		_x = data[:,1];		_y = data[:,5]

		if smooth:
			gauss_kernel = Gaussian1DKernel(sigma)
			plt.plot(_x, convolve(_y, gauss_kernel), lw='2' ,label=r'$x=${:.3f}'.format(contrt))
		else :
			plt.plot(_x, _y, lw='2' ,label=r'$x=${:.3f}'.format(contrt))

	plt.xlabel(r'$T$', fontsize=25)
	plt.ylabel(r'$E/N$', fontsize=25)
	plt.xlim(xmin=2.0, xmax=6.0)
	plt.xticks(size=25)
	plt.yticks(size=25)
	#plt.legend(fontsize=18)
	plt.tight_layout()
	plt.show()


def magnetiza(names, smooth=False, sigma=5):
	
	for i, name in enumerate(names):
		data = np.loadtxt(name, dtype=float)
		num = float(name.split('.')[0])
		contrt = num/32.0
		_x = data[:,1];		_y = data[:,9]

		if smooth:
			gauss_kernel = Gaussian1DKernel(sigma)
			plt.plot(_x, convolve(_y, gauss_kernel), lw='2' ,label=r'$x=${:.3f}'.format(contrt))
		else :
			plt.plot(_x, _y, lw='2' ,label=r'$x=${:.3f}'.format(contrt))

	plt.xlabel(r'$T$', fontsize=25)
	plt.ylabel(r'$M/N$', fontsize=25)
	plt.xlim(xmin=2.0, xmax=6.0)
	plt.xticks(size=25)
	plt.yticks(size=25)
	#plt.legend(fontsize=18)
	plt.tight_layout()
	plt.show()



def main():

	names = []
	for k in sys.argv[1:]:
		names.append(k + '.out')

	heat_capacity(names)
	heat_capacity(names, smooth=True)
	heat_capacity_per_T(names)
	heat_capacity_per_T(names, smooth=True)
	susceptibility(names)
	susceptibility(names, smooth=True)
	energy(names)
	energy(names, smooth=True)
	magnetiza(names)
	temperature()

if __name__ == '__main__':
	main()
