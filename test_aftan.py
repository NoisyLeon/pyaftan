import pyaftan
import obspy
# tr=obspy.read('/home/lili/code/SES3DPy/SES.118S28.SAC')[0]
# tr=obspy.read('./sac_data/SES.98S46.SAC')[0]
tr=obspy.read('./test.sac')[0]
tr=obspy.read('../SES3DPy/SES.103S23.SAC')[0]
atr=pyaftan.aftantrace(tr.data, tr.stats)
# for i in xrange(1000):
#     print i
# d1=atr.aftan(tmin=8., tmax=20.)
# atr.aftanf77(tmin=8., tmax=20.)
d1=atr.aftan(tmin=1., tmax=100., phvelname='ak135.disp')
atr.aftanf77(tmin=1., tmax=100., phvelname='ak135.disp')