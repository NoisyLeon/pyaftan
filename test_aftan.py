import pyaftan
import obspy
import matplotlib.pyplot as plt
import numpy as np



tr=obspy.read('./sac_data/SES.98S47.SAC')[0]
atr1=pyaftan.aftantrace(tr.data, tr.stats)
atr2=pyaftan.aftantrace(tr.data, tr.stats)
# aftan analysis using pyaftan
d1=atr1.aftan(tmin=5., tmax=30., vmin=2.5, vmax=4.5, phvelname='ak135.disp')


# atr1.plotftan(plotflag=3)
# plt.suptitle('pyaftan results')
# aftan analysis using compiled fortran library
atr2.aftanf77(tmin=5., tmax=30., vmin=2.5, vmax=4.5, phvelname='ak135.disp')
# atr2.plotftan(plotflag=3)
# plt.suptitle('fortran77 aftan results')
plt.show()