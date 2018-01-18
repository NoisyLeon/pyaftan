import pyaftan
import obspy
import matplotlib.pyplot as plt
import numpy as np



tr=obspy.read('./test.sac')[0]
atr=pyaftan.aftantrace(tr.data, tr.stats)
atr.aftanf77(tmin=2., tmax=40., phvelname='ak135.disp')

f   = 1./15.
filtered_tr     = atr.gaussian_filter_aftan(f)

plt.plot(np.arange(atr.stats.npts)*atr.stats.delta, filtered_tr)
plt.show()