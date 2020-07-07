import pyaftan
import obspy
import matplotlib.pyplot as plt
import numpy as np



tr=obspy.read('./COR_TA.M14A_TA.M17A.SAC')[0]
atr1=pyaftan.aftantrace(tr.data, tr.stats)
atr2=pyaftan.aftantrace(tr.data, tr.stats)
atr1.makesym()
atr2.makesym()
# aftan analysis using pyaftan
atr1.aftan(tmin=2., tmax=40., phvelname='ak135.disp')
atr1.plotftan(plotflag=3)
# plt.suptitle('pyaftan results')
# aftan analysis using compiled fortran library
atr2.aftanf77(tmin=2., tmax=40., phvelname='ak135.disp')
atr2.plotftan(plotflag=3)
plt.suptitle('fortran77 aftan results')
plt.show()
