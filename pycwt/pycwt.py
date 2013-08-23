'''
Python Continuous Wavelet Transform
===================================

This module implements the Continuous Wavelet Transform (CWT).
Mathematics are taken from Torrence and Compo 1998, and this Python
code is a significantly refactored (though (hopefully!) mathematically
identical) version of their Matlab original.

The following copyright notice appears in the original:

--------------------------------------------------------------------------
  Copyright (C) 1995-2004, Christopher Torrence and Gilbert P. Compo

  This software may be used, copied, or redistributed as long as it is not
  sold and this copyright notice is reproduced on each copy made. This
  routine is provided as is without any express or implied warranties
  whatsoever.

Notice: Please acknowledge the use of the above software in any publications:
   ``Wavelet software was provided by C. Torrence and G. Compo,
     and is available at URL: http://paos.colorado.edu/research/wavelets/.

Reference: Torrence, C. and G. P. Compo, 1998: A Practical Guide to
           Wavelet Analysis. <I>Bull. Amer. Meteor. Soc.</I>, 79, 61-78.

Please send a copy of such publications to either C. Torrence or G. Compo:
 Dr. Christopher Torrence               Dr. Gilbert P. Compo
 Research Systems, Inc.                 Climate Diagnostics Center
 4990 Pearl East Circle                 325 Broadway R/CDC1
 Boulder, CO 80301, USA                 Boulder, CO 80305-3328, USA
 E-mail: chris[AT]rsinc[DOT]com         E-mail: compo[AT]colorado[DOT]edu
--------------------------------------------------------------------------

A number of useful functions have been added, including cross-wavelet analysis,
a non-parametric significance testing procedure, functions for time- and
scale-averageing, and plotting methods.

'''

import scipy as sp
from scipy.stats.distributions import chi2
from scipy import special
from scipy.special import gamma
from scipy.ndimage import convolve
from correlations import acvf
import matplotlib.pyplot as plt
from matplotlib.mlab import fftsurr

class Wavelet(object):
    '''
    Object representing a mother wavelet.
    '''
    def __init__(self, fourier_factor, dofmin, coi_factor, C_delta, gamma, dj0):
        self.fourier_factor = fourier_factor
        self.dofmin = dofmin
        self.coi_factor = coi_factor
        self.C_delta = C_delta
        self.gamma = gamma
        self.dj0 = dj0
    
    def daughter(self, scale, N, dt=1):
        pass
    

class Paul(Wavelet):
    def __init__(self, order=4):
        '''
        Object representing the mother Paul wavelet.
        
        Parameters
        ----------
        (for the __init__ method)
        order : int
            Shape parameter for the wavelet, defaults to 4.
        '''
        self.order = order
        if order == 4:
            Wavelet.__init__(self,
                        fourier_factor=4 * sp.pi / (2 * order + 1),
                        dofmin=2, coi_factor=1 / sp.sqrt(2), C_delta=1.132,
                        gamma=1.17, dj0=1.5)
        else:
            Wavelet.__init__(self,
                        fourier_factor=4 * sp.pi / (2 * order + 1),
                        dofmin=2, coi_factor=1 / sp.sqrt(2), C_delta=None,
                        gamma=None, dj0=None)
    
    def daughter(self, scale, N, dt=1):
        k = sp.arange(int(N/2)) * 2 * sp.pi / (N * dt)
        k = sp.hstack((k, -sp.flipud(k)))
        if len(k) == N + 1:
            k = k[1: ]
    	expnt = -(scale * k) * (k > 0.)
    	norm = sp.sqrt(scale * k[1]) * (2**self.order 
    	    / sp.sqrt(self.order * sp.prod(sp.arange(2, (2 * self.order - 1))))) * sp.sqrt(N)
    	daughter = norm * ((scale * k)**self.order) * sp.exp(expnt);
    	daughter = daughter * (k > 0.)     # Heaviside step function
    	return daughter

class DOG(Wavelet):
    def __init__(self, order=2):
        '''
        Object representing the mother derivative-of-Gaussian (DOG or 'Mexican Hat')
        wavelet.
        
        Parameters
        ----------
        (for the __init__ method)
        order : int
            Shape parameter for the wavelet, currently only accepts order 2.
        '''
        self.order = order
        if order == 2:
            Wavelet.__init__(self, 
                        fourier_factor = 2 * sp.pi * sp.sqrt(2. / (2 * order + 1)),
                        dofmin=1, coi_factor=sp.sqrt(2), C_delta=3.541, gamma=1.43,
                        dj0=1.4)
        elif order == 6:
            Wavelet.__init__(self, 
                        fourier_factor = 2 * sp.pi * sp.sqrt(2. / (2 * order + 1)),
                        dofmin=1, coi_factor=sp.sqrt(2), C_delta=1.966, gamma=1.37,
                        dj0=0.97)
        else:
            Wavelet.__init__(self, 
                        fourier_factor = 2 * sp.pi * sp.sqrt(2. / (2 * order + 1)),
                        dofmin=1, coi_factor=sp.sqrt(2), C_delta=None, gamma=None,
                        dj0=None)
    
    def daughter(self, scale, N, dt=1):
        k = sp.arange(int(N/2)) * 2 * sp.pi / (N * dt)
        k = sp.hstack((k, -sp.flipud(k)))
        if len(k) == N + 1:
            k = k[1: ]
        expnt = -(scale * k)**2 / 2.0
    	norm = sp.sqrt(scale * k[1] / special.gamma(self.order + 0.5)) * sp.sqrt(N);
    	daughter = -norm * (1j**self.order) * ((scale * k)**self.order) * sp.exp(expnt)
    	return daughter 
	
    

class Morlet(Wavelet):
    def __init__(self, k0=6):
        '''
        Object representing the mother Morlet wavelet.
        
        Parameters
        ----------
        (for the __init__ method)
        k0 : int
            Frequency parameter, defaults to 6.
        '''
        self.k0 = k0
        Wavelet.__init__(self,
            fourier_factor=(4*sp.pi) / (self.k0+sp.sqrt(2+self.k0**2)),
            dofmin=2, coi_factor=1/sp.sqrt(2), C_delta=0.776, gamma=2.32, dj0=0.6)
    
    def daughter(self, scale, N, dt=1):
        '''
        Returns a daughter wavelet to be multiplied with the
        Fourier-transformed time series.
        
        Parameters
        ----------
        scale : float
            Scale of the wavelet.
        N : int
            Number of observations in the series being transformed.
        dt : int
            Number of observations per unit time.
            
        Returns
        -------
        daughter : ndarray
            "Daughter" wavelet (the Fourier transform of the
            mother wavelet of the appropriate scale and length)
        '''
        k = sp.arange(int(N/2)) * 2 * sp.pi / (N * dt)
        k = sp.hstack((k, -sp.flipud(k)))
        if len(k) == N + 1:
            k = k[1: ]
        expnt = -(scale * k - self.k0)**2 / 2. * (k > 0)
        norm = sp.sqrt(scale * k[1]) * sp.pi**(-0.25) * sp.sqrt(N) # total energy = N (Eqn. 7)
        daughter = norm * sp.exp(expnt)
        daughter = daughter * (k >0)
        return daughter   
    

class WaveletTransform(object):
    '''
    Object encapsulating the results of a continuous wavelet transform.
    
    Parameters
    ----------
    (For __init__ method)
    wave : complex ndarray
        Array of the wavelet coefficients.
    scales : ndarray
        The scales at which the transform was performed.
    dscale : float
        The resolution in scale (i.e., fractions of an octave)
    wavelet : Wavelet
        Mother Wavelet object to use in the transform.
    period : ndarray
        Array of physical periods the transform was performed at.
        Equal to `scales` multiplied by `wavelet.fourier_factor`.
    series : ndarray
        The series the transform was performed on.
    dt : float
        Sampling interval of the time series.
    coi : ndarray
        Cone of influence.  CWT coefficients outside this area
        are likely decreased in magnitude due to zero-padding
        at the ends of the series.
    '''
    def __init__(self, series, wave, scales, dscale, wavelet, dt=1.0):
        self.series = series
        self.time = sp.arange(len(series)) * dt
        self.wave = wave
        self.scales = scales
        self.dscale = dscale
        self.wavelet = wavelet
        self.period = self.scales * self.wavelet.fourier_factor
        self.dt = dt
        self.coi = wavelet.coi_factor # * wavelet.fourier_factor
    
    def power(self):
        '''
        Return the array of wavelet power (i.e., the squared modulus
        of the complex wavelet coefficients).
        '''
        return abs(self.wave)**2
    
    def phase(self, degrees=False):
        '''
        Return an array of the phase angles of the wavelet coefficients,
        in radians (set degrees=True for degrees).
        '''
        phase = sp.angle(self.wave)
        if degrees:
            phase *= 180 / sp.pi
        return phase
    
    def _sig_surface(self, siglevel):
        '''
        Significance surface for plotting.
        '''
        sig = wave_signif(self, siglevel, lag1(self.series))
        sig = sp.tile(sig, (len(self.series), 1)).T
        return sig
    
    def _add_coi(self, color, data_present=None, fill=False):
        n = len(self.series)
        coi_whole = self.coi * self.dt * sp.hstack((sp.arange((n + 1) / 2), 
                            sp.flipud(sp.arange(n / 2))))
        coi_list = [coi_whole]
        baseline = sp.ones(n) * self.period[-1]
        if data_present is not None:
            for i in range(2, len(data_present) - 1):
                if data_present[i - 1] and (not data_present[i]):
                    coi_list.append(circ_shift(coi_whole, i))
                    baseline[i] = 0
                elif not data_present[i]:
                    baseline[i] = 0
                elif (not data_present[i - 1]) and data_present[i]:
                    coi_list.append(circ_shift(coi_whole, i))
        coi_list.append(baseline)
        coi_line = sp.array(coi_list).min(axis=0)
        coi_line[coi_line == 0] = 1e-4
        x = sp.hstack((self.time, sp.flipud(self.time)))
        y = sp.log2(sp.hstack((coi_line, sp.ones(n) * self.period[-1])))
        if fill:
            plt.fill(x, y, color='black', alpha=0.3)
        else:
            plt.plot(self.time, sp.log2(coi_line), color=color, linestyle=':')
    
    def plot(self, norm=False, plot_type='power', degrees=False, kernel=None, coi=True,
            data_present=None, siglevel=None, lag1=0.0, coi_color='white',
            coi_fill=False, sig_color='white', *args, **kwargs):
        '''
        Display an image of the wavelet coefficients or phase.
        
        Parameters
        ----------
        norm : logical
            If true, plot the wavelet power normalized by the time series
            variance. Defaults to False.
        plot_type : string
            Specifies what to plot.  Options are 'power', 'phase', and
            'coherence'. Defaults to 'power'
        degrees : logical
            If plotting the phase, should it be in degrees instead of 
            radians?  Defaults to False.
        kernel : array-like
            If plotting the coherence, a smoothing kernel must be supplied.
        data_present : array-like
            Boolean array the same length as the time series, with "True" values
            where corresponding to valid measurements.  Used to draw cone-of-
            influence lines in the middle of the series (as in the case of
            missing of invalid data).
        siglevel : float
            If given, plots the significance contours at this level.
        lag1 : float
            Lag-1 autocorrelation used for significance testing using an
            AR(1) red-noise null hypothesis.
        coi_color, sig_color : matplotlib.mpl.colors colors
            colors for the cone-of-influence and significance
            contour, if plotted.  Default to white.
        coi_fill : logical
            If true, plot the coi as a transparently-shaded region. Defaults to
            False.
        *args, **kwargs : Additional arguments passed to contourf().
        
        Returns
        -------
        A `matplotlib.image.AxesImage` instance.
        '''
        if plot_type == 'phase':
            values = self.phase(degrees=degrees)
            colormap = plt.cm.hsv
        elif plot_type == 'coherence':
            values = self.coherence(kernel)
            colormap = plt.cm.jet
        elif plot_type == 'power':
            values = self.power()
            colormap = plt.cm.jet
        else:
            raise ValueError("plot_type must be 'power', 'phase', or 'coherence'.")
        if norm:
            values /= self.series[sp.isfinite(self.series)].var()
        ax = plt.contourf(self.time, sp.log2(self.scales), values,
                *args, **kwargs)
        plt.ylim(sp.log2(sp.array([self.scales[-1], self.scales[0]])))
        yt = plt.yticks()
        #plt.yticks(yt[0] + 1, 2**yt[0]) 
        plt.yticks(yt[0], 2**yt[0]) 
        if coi:
            self._add_coi(color=coi_color, data_present=data_present)
        if siglevel is not None:
            if phase:
                print "Significance testing not available for wavelet phase."
            sig = self._sig_surface(siglevel)
            plt.contour(self.time, sp.log2(self.period), self.power() - sig, 
                    levels=[0], colors=sig_color, antialiased=True)   
        return ax



class CrossWaveletTransform(WaveletTransform):
    def __init__(self, series1, series2, wave1, wave2, *args, **kwargs):
        series = sp.vstack((series1.ravel(), series2.ravel())).T
        wave = sp.dstack((wave1, wave2))
        WaveletTransform.__init__(self, series, wave, *args, **kwargs)
    
    def power(self):
        return abs(self.wave[:,:,0] * sp.conj(self.wave[:,:,1]))
        
    def coherence(self, kernel):
        numerator = convolve(self.power(), kernel)
        
        denominator = (convolve(abs(self.wave[:,:,0])**2, kernel)
                    * convolve(abs(self.wave[:,:,1])**2, kernel))**0.5,
        #denominator = convolve((abs(self.wave[:,:,0])**2 * abs(self.wave[:,:,1])**2)**0.5,
        #                kernel)
                               
        return (numerator / denominator).reshape((self.scales.size, self.series.shape[0]))
    
    def _sig_surface(self, siglevel):
        a1 = lag1(self.series[:, 0])
        a2 = lag1(self.series[:, 1])
        sig = cross_wave_signif(self, siglevel, a1, a2)
        return sp.tile(sig, (len(self.series), 1)).T
        
        
def cwt(series, wavelet, octaves=None, dscale=0.25, minscale=None, dt=1.0):
    '''
    Perform a continuous wavelet transform on a series.
        
    Parameters
    ----------
    series : ndarray
    octaves : int
        Number of powers-of-two over which to perform the transform.
    dscale : float
        Fraction of power of two separating the scales. Defaults to 0.25.
    minscale : float
        Minimum scale. If none supplied, defaults to 2.0 * dt.
    dt : float
        Time step between observations in the series.
        
    Returns
    -------
    WaveletTransform
        WaveletTransform object with the results of the CWT.
    
    See Also
    --------
    ccwt : Cross continuous wavelet transform, for the wavelet
        coherence between two series
    
    Notes
    -----
    This function uses a fast Fourier Transform (FFT) to convolve
    the wavelet with the series at each scale.  For details, see:
    
    Torrence, C. and G. P. Compo, 1998: A Practical Guide to
    Wavelet Analysis. <I>Bull. Amer. Meteor. Soc.</I>, 79, 61-78.
    '''
    # Generate the array of scales
    if not minscale: minscale = 2.0 * dt
    if not octaves:
        octaves = int(sp.log2(len(series) * dt / minscale) / dscale) * dscale
    scales = minscale * 2**sp.arange(octaves + dscale, step=dscale)
    # Demean and pad time series with zeroes to next highest power of 2
    N = len(series)
    series = pad(series - series.mean())
    N_padded = len(series)
    wave = sp.zeros((len(scales), N_padded)) + complex(0, 0)
    series_ft = sp.fft(series)
    for i, s in enumerate(scales):
        wave[i, :] = sp.ifft(series_ft * wavelet.daughter(s, N_padded, dt))
    wave = wave[:, :N]
    series = series[:N]
    return WaveletTransform(series, wave, scales, dscale, wavelet, dt)

def ccwt(series1, series2, *args, **kwargs):
    t1 = cwt(series1, *args, **kwargs)
    t2 = cwt(series2, *args, **kwargs)
    return CrossWaveletTransform(t1.series, t2.series, t1.wave, t2.wave, 
            t2.scales, t2.dscale, t2.wavelet, t2.dt)

def pad(series):
    '''
    Returns a time series padded with zeros to the
    next-highest power of two.
    '''
    N = len(series)
    next_N = 2 ** sp.ceil(sp.log2(N))
    return sp.hstack((series, sp.zeros(next_N - N)))

def normalize(series):
    '''
    Returns the series demeaned and divided by its standard deviation.
    '''
    mean = series[sp.isfinite(series)].mean()
    sdev = series[sp.isfinite(series)].std()
    return (series - mean) / sdev

def add_half_coi(cwt, t, coi, edge):
    '''
    edge : int
        If edge = 1, add the COI to the right of the current tumn.
        If edge = -1, add the coi to the left of the current tumn.
    '''
    for s in sp.arange(cwt.shape[0]):
        bounds = [t, t + edge * coi[s]]
        cwt[s, min(bounds):max(bounds) + 1] = True

def mask_coi(cwt, coi, data_present=None, axis=0):
    '''
    mask_coi(cwt, coi, series=None, axis=0)
    
    Return a copy of cwt, masked with cone of influence coi.
    
    Parameters
    ----------
    cwt : ndarray
        An array of wavelet transform coefficients.
    coi : array_like
        Widths of the cone of influence, as a function of scale.
        Length must match the scale dimension of the cwt array.
    data_present : array_like, optional
        Optional series of boolean values, representing the locations
        of missing data in the original time series.  Values of 0 or
        False indicate missing values.  If supplied, this series will
        be used to draw cones of influence at the edges of the missing data
        regions.
    axis : integer, optional
        Axis along which the COI is added (i.e., the scale axis). Defaults
        to 0.
    
    Returns
    -------
    masked_array(cwt, mask)
        The input array, masked by the appropriate cone(s) of influence
    '''
    if axis == 1:
        cwt = cwt.T
    mask = sp.zeros_like(abs(cwt))
    add_half_coi(mask, 0, coi, 1)
    if data_present != None:
        for i in range(2, len(data_present) - 1):
            if data_present[i - 1] and (not data_present[i]):
                add_half_coi(mask, i, coi, -1)
            elif (not data_present[i - 1]) and data_present[i]:
                add_half_coi(mask, i, coi, 1)
            elif not data_present[i]:
                mask[: , i] = True
    add_half_coi(mask, mask.shape[1], coi, -1)
    return sp.ma.masked_array(cwt, mask)

def phase(wave):
    return sp.arctan(sp.imag(wave) / sp.real(wave))


def circ_shift(x, shift):
    return sp.hstack((x[-shift: ], x[:-shift]))

def lag1(x):
    '''
    Find the lag-1 autocorrelation of a time series (i.e. fit an AR(1) model).
    Uses the Levinson-Durbin algorithm.
    '''
    a = acvf(x)
    return a[1] / a[0]

def red_spectrum(lag1, freq):
    return (1. - lag1**2) / (1.0 - 2.0 * lag1 * sp.cos(freq * 2.0 * sp.pi) + lag1**2)

def d_cross_dist_nonvector(z, dof):
    '''
    Probability density function for cross-wavelet spectrum, assuming
    both wavelet spectra are chi-squared distributed.  From equation (30)
    in Torrence and Compo, 1998.
    
    Parameters
    ----------
    z : float
        The random variable.
    dof : int
        Degrees of freedom of the wavelet (1 for real wavelets, 2 for complex ones).
        
    Returns
    -------
    d : float
        Probability density at z.
    '''
    if z == 0:
        return 0.
    else:
        return (2.**(2 - dof) / special.gamma(dof / 2)**2 
                * z**(dof - 1) * special.k0(z))

d_cross_dist = sp.vectorize(d_cross_dist_nonvector)


@sp.vectorize
def p_cross_dist(q, dof):
    return sp.integrate.quad(d_cross_dist_nonvector, 0, q, (dof,))[0]

@sp.vectorize
def q_cross_dist(p, dof):
    objective = lambda q: abs(p - p_cross_dist(q, dof))
    return sp.optimize.fmin(objective, 1, disp=0)

def cross_wave_signif(t, siglevel=0.95, lag1=0.0, lag2=0.0):
    dof = t.wavelet.dofmin
    std1, std2 = sp.std(t.series, axis=0)
    fft_theor1 = red_spectrum(lag1, t.dt / t.period)
    fft_theor2 = red_spectrum(lag2, t.dt / t.period)
    q = q_cross_dist(siglevel, dof)
    return std1 * std2 * sp.sqrt(fft_theor1 * fft_theor2) * q / dof
    
def wave_signif(t, siglevel=0.95, lag1=0.0, test='local', dof=None, scale_range=None):
    fft_theor = red_spectrum(lag1, t.dt / t.period)
    fft_theor *= t.series.var()  # Include time-series variance
    # No smoothing, DOF = dofmin (Sec. 4)
    if test == 'local':
        return fft_theor * chi2.ppf(siglevel, t.wavelet.dofmin) / t.wavelet.dofmin
    # Time-averaged significance
    elif test == 'global':
        # Eqn. 23
        dof = t.wavelet.dofmin * sp.sqrt(1 + ((len(t.series) * t.dt) / 
                                        (t.wavelet.gamma * t.scales))**2)
        dof[dof < t.wavelet.dofmin] = t.wavelet.dofmin # minimum DOF is dofmin
        return fft_theor * chi2.ppf(siglevel, dof) / dof
    elif test == 'scale':
        if not scale_range:
            raise ValueError("Must supply a scale_range for time-averaged \
                            significance testing.")
        if period:
            scale_indices = (transform.period >= min(scale_range)) \
                            & (transform.period <= max(scale_range))
        else:
            scale_indices = (transform.scales >= min(scale_range)) \
                            & (transform.scales <= max(scale_range))  
        scale_indices = sp.arange(len(scale_indices))[scale_indices]
        na = len(t.series)
        savg = scale_avg(t, min(scale_range), max(scale_range))
        smid = t.minscale * 2 ** (0.5 * (min(scale_range) + max(scale_range)) * t.dscale)
        dof = 2 * len(na) * savg[1] / smid \
                * sp.sqrt(1 + (na * t.dscale / t.wavelet.dj0)**2)
        P = savg[1] * (fft_theor[scale_indices] / t.scales[scale_indices]).sum()

def time_avg(transform, start=None, end=None):
    '''
    Doesn't work as intended for cross-wavelet transforms'
    '''
    return (abs(transform.wave[:, start:end])**2).mean(axis=1) / sp.var(transform.series)

def scale_avg(transform, min, max, period=True, norm=False):
    if period:
        scale_indices = (transform.period >= min) & (transform.period <= max)
    else:
        scale_indices = (transform.scales >= min) & (transform.scales <= max)  
    scale_indices = sp.arange(len(scale_indices))[scale_indices]      
    band = transform.wave[scale_indices, :]
    scales = transform.scales[scale_indices].reshape((len(scale_indices), 1))
    w = transform.wavelet
    W_avg = (transform.dscale * transform.dt / w.C_delta) \
                * (abs(band)**2 / scales).sum(axis=0) # Eqn. 24
    scale_avg = float(1 / sum(1 / scales))
    if norm:
        W_avg = W_avg * t.wavelet.C_delta * scale_avg / (t.dscale * t.dt * t.series.var())
    return W_avg, scale_avg

def plot_cwt(t):
    s1 = plt.subplot(221)
    t.plot()
    s2 = plt.subplot(222)
    spec = time_avg(t)
    plt.plot(spec, sp.log2(t.period))
    plt.ylim(sp.log2(t.period).max(), sp.log2(t.period).min())
    nscales = len(t.scales)
    yt = sp.arange(nscales, step=int(1 / self.dscale))
    plt.yticks(yt, t.scales[yt])
    plt.ylim(nscales - 1, 0)
    s1.set_position((0.1, 0.1, 0.65, 0.8))
    s2.set_position((0.8, 0.1, 0.15, 0.8))


def bootstrap_signif(t, n):
    '''
    Estimates the significance level of a wavelet transform using a
    nonparametric bootstrapping procedure.
    
    Parameters
    ----------
    t : WaveletTransform or CrossWaveletTransform
    n : int
        Number of realizations of the random series to test.
    
    Returns
    -------
    signif : array
        Float array the same shape as t.wave, containing estimated p-values
        for each time and scale.
    
    Details
    -------
    Generates n simulated time series with the same power spectrum as
    the original series, via phase randomization, performing the wavelet
    transform on each one.  For each time/scale point in the original
    transform's wavelet spectrum, counts the number of times the 
    corresponding point in a simulated spectrum is greater, then divides
    by the number of simulations to get the p-value.
        
    '''
    n_greater = sp.zeros_like(t.power())
    if type(t) == CrossWaveletTransform:
        for i in range(n):
            x1 = fftsurr(t.series[:, 0])
            x2 = fftsurr(t.series[:, 1])
            t_sim = ccwt(x1, x2, t.wavelet, octaves=(t.wave.shape[0]-1) * t.dscale,
                        dscale=t.dscale, minscale=t.scales.min(), dt=t.dt)
            n_greater += t_sim.power() > t.power()
    else:
        for i in range(n):
            x = fftsurr(t.series)
            t_sim = cwt(x, t.wavelet, octaves=(t.wave.shape[0]-1) * t.dscale,
                        dscale=t.dscale, minscale=t.scales.min(), dt=t.dt)
            n_greater += t_sim.power() > t.power()
    return n_greater / n

