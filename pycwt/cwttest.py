import pycwt
import numpy as np
from pylab import *

filename = 'sst_nino3.dat'
data = loadtxt(filename)

# remove mean
data = (data - np.nansum(data) / len(data))
data[np.isnan(data)] = 0

t = pycwt.cwt(data, pycwt.Morlet(), octaves=8, dscale=0.1)

b = pycwt.bootstrap_signif(t, 200)
imshow(t.power(), aspect='auto')
contour(b, levels=[0.05], colors='w')
figure()
plot(pycwt.time_avg(t), t.scales)