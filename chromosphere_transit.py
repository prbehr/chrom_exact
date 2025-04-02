import numpy as np

def ellk(k):
	# Computes the polynomial approximation for the complete elliptic integral 
	# of the first kind (Hasting's approximation)
	m1 = 1-k**2
	a0 = 1.38629436112
	a1 = 0.09666344259
	a2 = 0.03590092383
	a3 = 0.03742563713
	a4 = 0.01451196212
	b0 = 0.50000000000
	b1 = 0.12498593597
	b2 = 0.06880248576
	b3 = 0.03328355346
	b4 = 0.00441787012

	ek1 = a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
	ek2 = (b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*np.log(m1)
	
	return ek1-ek2


def ellc(k):
	# Computes polynomial approximation for the complete elliptic
	# integral of the second kind (Hasting's approximation):
	m1=1-k**2
	a1=0.44325141463
	a2=0.06260601220
	a3=0.04757383546
	a4=0.01736506451
	b1=0.24998368310
	b2=0.09200180037
	b3=0.04069697526
	b4=0.00526449639
	ee1=1+m1*(a1+m1*(a2+m1*(a3+m1*a4)))
	ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*np.log(1/m1)
	return ee1+ee2


def ellpic_bulirsch(n,k):
	# Computes the complete elliptical integral of the third kind using
	# the algorithm of Bulirsch (1965):
	kc=np.sqrt(1-k**2)
	p=n+1

	if(min(p) < 0):
		print('Negative p')

	m0 = 1
	c = 1
	p = np.sqrt(p)
	d = 1/p
	e = kc

	converged = False
	while(not converged):
		f = c
		c = d/p+f
		g = e/p
		d = (f*g+d)*2
		p = g+p
		g = m0
		m0 = kc+m0

		if np.max(np.abs(1-kc/g)) > 1e-13:
			kc = 2*np.sqrt(e)
			e = kc*m0
		else:
			converged = True

	return 0.5*np.pi*(c*m0+d)/(m0*(m0+p))

def chrom_exact(b,p):
	# Please cite Schlawin et al. (2010) if you make use of this IDL code
	#
	# IDL code for computing the transit of a chromosphere by a spherical
	# planet.
	#
	# Input:
	#   b     vector of impact parameter values in units of the stellar radius
	#   p     R_p/R_* = ratio of planet radius to stellar radius
	# Output:
	#   lc    light curve of a chromospheric transit.
	# 

	a = np.zeros(len(b))
	indx = np.where(b+p < 1)[0]

	if len(indx) > 0:
		k = np.sqrt(4.0*b[indx]*p/(1-(b[indx]-p)**2))
		a[indx]=4.0/np.sqrt(1.0-(b[indx]-p)**2)*(((b[indx]-p)**2-1.0)*ellc(k)
             -(b[indx]**2-p**2)*ellk(k)+(b[indx]+p)/(b[indx]-p)
             *ellpic_bulirsch(4.0*b[indx]*p/(b[indx]-p)**2,k))

	indx = np.where((b+p > 1)&(b-p < 1))[0]

	if len(indx) > 0:
		k = np.sqrt((1-(b[indx]-p)**2)/4/b[indx]/p)
		a[indx] = 2/(b[indx]-p)/np.sqrt(b[indx]*p)*\
			(4*b[indx]*p*(p-b[indx])*ellc(k)+
			(-b[indx]+2*b[indx]**2*p+p-2*p**3)*ellk(k)+
			(b[indx]+p)*ellpic_bulirsch(-1+1/(b[indx]-p)**2,k)
			)

	lc = 1-(4*np.pi*(p>b)+a)/(4*np.pi)

	return lc