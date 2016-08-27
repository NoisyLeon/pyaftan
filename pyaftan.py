import obspy
import numpy as np
import aftan
import scipy.interpolate
import matplotlib.pyplot as plt
import os
import warnings
from scipy.signal import argrelmax, argrelmin
import scipy.interpolate 
try:
    import pyfftw
    useFFTW=True
except:
    useFFTW=False

class ftanParam(object):
    """ An object to handle ftan output parameters
    ===========================================================================
    Basic FTAN parameters:
    nfout1_1 - output number of frequencies for arr1, (integer*4)
    arr1_1   - preliminary results.
                Description: real*8 arr1(8,n), n >= nfin)
                arr1_1[0,:] -  central periods, s
                arr1_1[1,:] -  observed periods, s
                arr1_1[2,:] -  group velocities, km/s
                arr1_1[3,:] -  phase velocities, km/s or phase if nphpr=0, rad
                arr1_1[4,:] -  amplitudes, Db
                arr1_1[5,:] -  discrimination function
                arr1_1[6,:] -  signal/noise ratio, Db
                arr1_1[7,:] -  maximum half width, s
                arr1_1[8,:] -  amplitudes
    arr2_1   - final results with jump detection
    nfout2_1 - output number of frequencies for arr2, (integer*4)
                Description: real*8 arr2(7,n), n >= nfin)
                If nfout2 == 0, no final result.
                arr2_1[0,:] -  central periods, s
                arr2_1[1,:] -  observed periods, s
                arr2_1[2,:] -  group velocities, km/sor phase if nphpr=0, rad
                arr2_1[3,:] -  phase velocities, km/s
                arr2_1[4,:] -  amplitudes, Db
                arr2_1[5,:] -  signal/noise ratio, Db
                arr2_1[6,:] -  maximum half width, s
                arr2_1[7,:] -  amplitudes
    tamp_1   -  time to the beginning of ampo table, s (real*8)
    nrow_1   -  number of rows in array ampo, (integer*4)
    ncol_1   -  number of columns in array ampo, (integer*4)
    amp_1    -  Ftan amplitude array, Db, (real*8)
    ierr_1   - completion status, =0 - O.K.,           (integer*4)
                                 =1 - some problems occures
                                 =2 - no final results
    ----------------------------------------------------------------------------
    Phase-Matched-Filtered FTAN parameters:
    nfout1_2 - output number of frequencies for arr1, (integer*4)
    arr1_2   - preliminary results.
                Description: real*8 arr1(8,n), n >= nfin)
                arr1_2[0,:] -  central periods, s (real*8)
                arr1_2[1,:] -  apparent periods, s (real*8)
                arr1_2[2,:] -  group velocities, km/s (real*8)
                arr1_2[3,:] -  phase velocities, km/s (real*8)
                arr1_2[4,:] -  amplitudes, Db (real*8)
                arr1_2[5,:] -  discrimination function, (real*8)
                arr1_2[6,:] -  signal/noise ratio, Db (real*8)
                arr1_2[7,:] -  maximum half width, s (real*8)
                arr1_2[8,:] -  amplitudes 
    arr2_2   - final results with jump detection
    nfout2_2 - output number of frequencies for arr2, (integer*4)
                Description: real*8 arr2(7,n), n >= nfin)
                If nfout2 == 0, no final results.
                arr2_2[0,:] -  central periods, s (real*8)
                arr2_2[1,:] -  apparent periods, s (real*8)
                arr2_2[2,:] -  group velocities, km/s (real*8)
                arr2_2[3,:] -  phase velocities, km/s (real*8)
                arr2_2[4,:] -  amplitudes, Db (real*8)
                arr2_2[5,:] -  signal/noise ratio, Db (real*8)
                arr2_2[6,:] -  maximum half width, s (real*8)
                arr2_2[7,:] -  amplitudes
    tamp_2   -  time to the beginning of ampo table, s (real*8)
    nrow_2   -  number of rows in array ampo, (integer*4)
    ncol_2   -  number of columns in array ampo, (integer*4)
    amp_2    -  Ftan amplitude array, Db, (real*8)
    ierr_2   - completion status, =0 - O.K.,           (integer*4)
                                =1 - some problems occures
                                =2 - no final results
    ===========================================================================
    """
    def __init__(self):
        # Parameters for first iteration
        self.nfout1_1=0
        self.arr1_1=np.array([])
        self.nfout2_1=0
        self.arr2_1=np.array([])
        self.tamp_1=0.
        self.nrow_1=0
        self.ncol_1=0
        self.ampo_1=np.array([],dtype='float32')
        self.ierr_1=0
        # Parameters for second iteration
        self.nfout1_2=0
        self.arr1_2=np.array([])
        self.nfout2_2=0
        self.arr2_2=np.array([])
        self.tamp_2=0.
        self.nrow_2=0
        self.ncol_2=0
        self.ampo_2=np.array([])
        self.ierr_2=0
        # Flag for existence of predicted phase dispersion curve
        self.preflag=False
        self.station_id=None

    def writeDISP(self, fnamePR):
        """
        Write FTAN parameters to DISP files given a prefix.
        fnamePR: file name prefix
        _1_DISP.0: arr1_1
        _1_DISP.1: arr2_1
        _2_DISP.0: arr1_2
        _2_DISP.1: arr2_2
        """
        if self.nfout1_1!=0:
            f10=fnamePR+'_1_DISP.0'
            Lf10=self.nfout1_1
            outArrf10=np.arange(Lf10)
            for i in np.arange(7):
                outArrf10=np.append(outArrf10, self.arr1_1[i,:Lf10])
            outArrf10=outArrf10.reshape((8,Lf10))
            outArrf10=outArrf10.T
            np.savetxt(f10, outArrf10, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf')
        if self.nfout2_1!=0:
            f11=fnamePR+'_1_DISP.1'
            Lf11=self.nfout2_1
            outArrf11=np.arange(Lf11)
            for i in np.arange(6):
                outArrf11=np.append(outArrf11, self.arr2_1[i,:Lf11])
            outArrf11=outArrf11.reshape((7,Lf11))
            outArrf11=outArrf11.T
            np.savetxt(f11, outArrf11, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %8.3lf')
        if self.nfout1_2!=0:
            f20=fnamePR+'_2_DISP.0'
            Lf20=self.nfout1_2
            outArrf20=np.arange(Lf20)
            for i in np.arange(7):
                outArrf20=np.append(outArrf20, self.arr1_2[i,:Lf20])
            outArrf20=outArrf20.reshape((8,Lf20))
            outArrf20=outArrf20.T
            np.savetxt(f20, outArrf20, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %12.4lf %8.3lf')
        if self.nfout2_2!=0:
            f21=fnamePR+'_2_DISP.1'
            Lf21=self.nfout2_2
            outArrf21=np.arange(Lf21)
            for i in np.arange(6):
                outArrf21=np.append(outArrf21, self.arr2_2[i,:Lf21])
            outArrf21=outArrf21.reshape((7,Lf21))
            outArrf21=outArrf21.T
            np.savetxt(f21, outArrf21, fmt='%4d %10.4lf %10.4lf %12.4lf %12.4lf %12.4lf %8.3lf')
        return
    
    def writeDISPbinary(self, fnamePR):
        """
        Write FTAN parameters to DISP files given a prefix.
        fnamePR: file name prefix
        _1_DISP.0: arr1_1
        _1_DISP.1: arr2_1
        _2_DISP.0: arr1_2
        _2_DISP.1: arr2_2
        """
        f10=fnamePR+'_1_DISP.0'
        np.savez(f10, self.arr1_1, np.array([self.nfout1_1]) )
        f11=fnamePR+'_1_DISP.1'
        np.savez(f11, self.arr2_1, np.array([self.nfout2_1]) )
        f20=fnamePR+'_2_DISP.0'
        np.savez(f20, self.arr1_2, np.array([self.nfout1_2]) )
        f21=fnamePR+'_2_DISP.1'
        np.savez(f21, self.arr2_2, np.array([self.nfout2_2]) )
        return
    

    def FTANcomp(self, inftanparam, compflag=4):
        """
        Compare aftan results for two ftanParam objects.
        """
        fparam1=self
        fparam2=inftanparam
        if compflag==1:
            obper1=fparam1.arr1_1[1,:fparam1.nfout1_1]
            gvel1=fparam1.arr1_1[2,:fparam1.nfout1_1]
            phvel1=fparam1.arr1_1[3,:fparam1.nfout1_1]
            obper2=fparam2.arr1_1[1,:fparam2.nfout1_1]
            gvel2=fparam2.arr1_1[2,:fparam2.nfout1_1]
            phvel2=fparam2.arr1_1[3,:fparam2.nfout1_1]
        elif compflag==2:
            obper1=fparam1.arr2_1[1,:fparam1.nfout2_1]
            gvel1=fparam1.arr2_1[2,:fparam1.nfout2_1]
            phvel1=fparam1.arr2_1[3,:fparam1.nfout2_1]
            obper2=fparam2.arr2_1[1,:fparam2.nfout2_1]
            gvel2=fparam2.arr2_1[2,:fparam2.nfout2_1]
            phvel2=fparam2.arr2_1[3,:fparam2.nfout2_1]
        elif compflag==3:
            obper1=fparam1.arr1_2[1,:fparam1.nfout1_2]
            gvel1=fparam1.arr1_2[2,:fparam1.nfout1_2]
            phvel1=fparam1.arr1_2[3,:fparam1.nfout1_2]
            obper2=fparam2.arr1_2[1,:fparam2.nfout1_2]
            gvel2=fparam2.arr1_2[2,:fparam2.nfout1_2]
            phvel2=fparam2.arr1_2[3,:fparam2.nfout1_2]
        else:
            obper1=fparam1.arr2_2[1,:fparam1.nfout2_2]
            gvel1=fparam1.arr2_2[2,:fparam1.nfout2_2]
            phvel1=fparam1.arr2_2[3,:fparam1.nfout2_2]
            obper2=fparam2.arr2_2[1,:fparam2.nfout2_2]
            gvel2=fparam2.arr2_2[2,:fparam2.nfout2_2]
            phvel2=fparam2.arr2_2[3,:fparam2.nfout2_2]
        plb.figure()
        ax = plt.subplot()
        ax.plot(obper1, gvel1, '--k', lw=3) #
        ax.plot(obper2, gvel2, '-.b', lw=3)
        plt.xlabel('Period(s)')
        plt.ylabel('Velocity(km/s)')
        plt.title('Group Velocity Comparison')
        if (fparam1.preflag==True and fparam2.preflag==True):
            plb.figure()
            ax = plt.subplot()
            ax.plot(obper1, phvel1, '--k', lw=3) #
            ax.plot(obper2, phvel2, '-.b', lw=3)
            plt.xlabel('Period(s)')
            plt.ylabel('Velocity(km/s)')
            plt.title('Phase Velocity Comparison')
        return



def _aftan_gaussian_filter(alpha, omega0, ns, indata, ome):
    om2 = -(ome-omega0)*(ome-omega0)*alpha/omega0/omega0
    b=np.exp(om2)
    b[np.abs(om2)>=40.]=0.
    filterred_data=indata*b
    filterred_data[ns/2:]=0
    filterred_data[0]/=2
    filterred_data[ns/2-1]=filterred_data[ns/2-1].real+0.j
    return filterred_data
    
    
class aftantrace(obspy.core.trace.Trace):
    """
    ses3dtrace:
    A derived class inherited from obspy.core.trace.Trace. This derived class have a variety of new member functions
    """
    def init_ftanParam(self):
        """
        Initialize ftan parameters
        """
        self.ftanparam=ftanParam()
    def init_snrParam(self):
        """
        Initialize SNR parameters
        """
        self.SNRParam=snrParam()
    def reverse(self):
        """
        Reverse the trace
        """
        self.data=self.data[::-1]
        return
    
    def aftan(self, pmf=True, piover4=-1.0, vmin=1.5, vmax=5.0, tmin=4.0, \
        tmax=30.0, tresh=20.0, ffact=1.0, taperl=1.0, snr=0.2, fmatch=1.0, phvelname='', predV=np.array([])):
        """ (Automatic Frequency-Time ANalysis) aftan analysis:
        ===========================================================================================================
        Input Parameters:
        pmf        - flag for Phase-Matched-Filtered output (default: True)
        piover4    - phase shift = pi/4*piover4, for cross-correlation piover4 should be -1.0
        vmin       - minimal group velocity, km/s
        vmax       - maximal group velocity, km/s
        tmin       - minimal period, s
        tmax       - maximal period, s
        tresh      - treshold for jump detection, usualy = 10, need modifications
        ffact      - factor to automatic filter parameter, usualy =1
        taperl     - factor for the left end seismogram tapering, taper = taperl*tmax,    (real*8)
        snr        - phase match filter parameter, spectra ratio to determine cutting point for phase matched filter
        fmatch     - factor to length of phase matching window
        phvelname  - predicted phase velocity file name
        predV      - predicted phase velocity curve, period = predV[:, 0],  Vph = predV[:, 1]
        
        Output:
        self.ftanparam, a object of ftanParam class, to store output aftan results
        ===========================================================================================================
        References:
        Levshin, A. L., and M. H. Ritzwoller. Automated detection, extraction, and measurement of regional surface waves.
             Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkh?user Basel, 2001. 1531-1545.
        Bensen, G. D., et al. Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements.
             Geophysical Journal International 169.3 (2007): 1239-1260.
        """
        # preparing for data
        try:
            self.ftanparam
        except:
            self.init_ftanParam()
        try:
            dist=self.stats.sac.dist
        except:
            dist, az, baz=obspy.geodetics.base.gps2dist_azimuth(self.stats.sac.evla, self.stats.sac.evlo,
                                self.stats.sac.stla, self.stats.sac.stlo) # distance is in m
            self.stats.sac.dist=dist/1000.
            dist=dist/1000.
        nprpv = 0
        # phprper=np.zeros(300)
        # phprvel=np.zeros(300)
        if predV.size != 0:
            phprper=predV[:,0]
            phprvel=predV[:,1]
            nprpv = predV[:,0].size
            phprper=np.append( phprper, np.zeros(300-phprper.size) )
            phprvel=np.append( phprvel, np.zeros(300-phprvel.size) )
            self.ftanparam.preflag=True
        elif os.path.isfile(phvelname):
            # print 'Using prefile:',phvelname
            php=np.loadtxt(phvelname)
            phprper=php[:,0]
            phprvel=php[:,1]
            nprpv = php[:,0].size
            phprper=np.append( phprper, np.zeros(300-phprper.size) )
            phprvel=np.append( phprvel, np.zeros(300-phprvel.size) )
            self.ftanparam.preflag=True
        else:
            warnings.warn('No predicted dispersion curve for:'+self.stats.network+'.'+self.stats.station, UserWarning, stacklevel=1)
        ############################################
        nfin    = 64
        npoints = 5  #  only 3 points in jump
        perc    = 50.0 # 50 % for output segment
        ip      = 1.0
        ############################################
        # tempsac=self.copy()
        # length=len(tempsac.data)
        # if length>32768:
        #     warnings.warn('Length of seismogram is larger than 32768!', UserWarning, stacklevel=1)
        #     nsam=32768
        #     tempsac.data=tempsac.data[:nsam]
        #     tempsac.stats.e=(nsam-1)*tempsac.stats.delta+tb
        #     sig=tempsac.data
        # else:
        #     sig=np.append(tempsac.data, np.zeros( float(32768-tempsac.data.size) ) )
        #     nsam=int( float (tempsac.stats.npts) )### for unknown reasons, this has to be done, nsam=int(tempsac.stats.npts)  won't work as an input for aftan
        dt=self.stats.delta
        tb=self.stats.sac.b
        nsam=self.stats.npts
        # end of preparing data
        ###
        # alpha=ffact*20.*np.sqrt(dist/1000.)
        alpha=ffact*20.
        # number of samples for tapering, left and right end
        ntapb = int(round(taperl*tmax/dt))
        ntape = int(round(tmax/dt))
        omb = 2.0*np.pi/tmax
        ome = 2.0*np.pi/tmin
        # tapering seismogram
        nb = int(max(2, round((dist/vmax-tb)/dt)))
        tamp = (nb-1)*dt+tb
        ne = int(min(nsam, round((dist/vmin-tb)/dt)+1))
        nrow = nfin
        ncol = ne-nb+1
        tArr=np.arange(ne-nb+1)*dt
        vArr=dist/(tArr+tb)
        vArr[vArr==np.inf]=9999.
        tdata, ncorr=self.taper( max(nb, ntapb+1), min(ne, self.stats.npts-ntape), ntapb, ntape)
        
        # # # old_data=aftan.taper(max(nb, ntapb+1), min(ne, self.stats.npts-ntape), length,sig,ntapb,ntape)
        # # # return tdata, old_data[0]
    
        ns=max(1<<(ncorr-1).bit_length(), 2**12)  # different !!!
        domega = 2.*np.pi/ns/dt
        step =(np.log(omb)-np.log(ome))/(nfin -1)
        omegaArr=np.exp(np.log(ome)+np.arange(nfin)*step)
        perArr=2.*np.pi/omegaArr
        # FFT
        if useFFTW:
            fftdata=pyfftw.interfaces.numpy_fft.fft(tdata, ns)
        else:
            fftdata=np.fft.fft(tdata, ns)
        ome=np.arange(ns)*domega
        phaArr=np.zeros((ne+3-nb, nfin))
        ampo=np.zeros((ne+3-nb, nfin))
        amp=np.zeros((ne+3-nb, nfin))
        #  main loop by frequency
        for k in xrange(nfin):
            # fs, b = aftan.ftfilt(alpha, omegaArr[k], domega, ns,fftdata)
            # Gaussian filter
            filterS=_aftan_gaussian_filter(alpha=alpha, omega0=omegaArr[k], ns=ns, indata=fftdata, ome=ome)
            # return fs, filterS
            if useFFTW:
                filterT=pyfftw.interfaces.numpy_fft.ifft(filterS, ns)
            else:
                filterT=np.fft.ifft(filterS, ns)
            # need to multiply by 2 due to zero padding of negative frequencies
            # but NO NEED to divide by ns due to the difference of numpy style and FFTW style
            filterT=2.*filterT 
            phaArr[:,k]=np.arctan2(np.imag(filterT[nb-2:ne+1]), np.real(filterT[nb-2:ne+1]))
            ampo[:,k] = np.abs(filterT[nb-2:ne+1])
            amp[:,k] = 20.*np.log10(ampo[:,k])
        # normalization amp diagram to 100 Db with three decade cutting
        amax=amp.max()
        amp= amp+100.-amax
        amp[amp<40.]=40.
        # # # ind_lmax=argrelmax(amp, axis=0)
        # # # index_sort=np.argsort(ind_lmax[1])
        # # # ind_j=(ind_lmax[0])[index_sort]
        # # # ind_k=(ind_lmax[1])[index_sort]
        
        tim=np.zeros(nfin)
        tvis=np.zeros(nfin)
        ampgr=np.zeros(nfin)
        grvel=np.zeros(nfin)
        snr=np.zeros(nfin)
        wdth=np.zeros(nfin)
        phgr=np.zeros(nfin)
        ind_all=[]
        ipar_all=[]
        for k in xrange(nfin):
            ampk=amp[:,k]
            ampok=ampo[:,k]
            ind_localmax=argrelmax(ampk)[0]
            omega=omegaArr[k]
            if ind_localmax.size==0:
                ind_localmax=np.append(ind_localmax, ne+1-nb)
            ind_all.append(ind_localmax)
            dph, tm, ph, t=self._fmax(amp=ampk, pha=phaArr[:,k], ind=ind_localmax, om=omega, piover4=piover4)
            imax=tm.argmax()
            ipar=np.zeros((6, ind_localmax.size))
            ipar[0, :]  = (nb+ind_localmax-2.+t)*dt # note the difference with aftanf77, due to ind_localmax
            ipar[1, :]  = 2*np.pi*dt/dph
            ipar[2, :]  = tm
            ipar[5, :]  = ph
            if ind_localmax.size==1:
                lmindex=(ampok[:ind_localmax[0]]).argmin()
                rmindex=(ampok[ind_localmax[0]:]).argmin()
                lm=(ampok[:ind_localmax[0]])[lmindex]
                rm=(ampok[ind_localmax[0]:])[rmindex]
            else:
                splitArr=np.split(ampok, ind_localmax)
                minArr=np.array([])
                minindexArr=np.array([])
                for tempArr in splitArr:
                    temp_ind_min=tempArr.argmin()
                    minArr=np.append(minArr, tempArr[temp_ind_min])
                    minindexArr=np.append(minindexArr, temp_ind_min)
                lm=minArr[:-1]
                rm=minArr[1:]
                minindexArr[1:]=minindexArr[1:]+ind_localmax
                lmindex=minindexArr[:-1]
                rmindex=minindexArr[1:]
            ipar[3,:] = 20.*np.log10(ampok[ind_localmax]/np.sqrt(lm*rm))
            ipar[4,:] = (np.abs(ind_localmax-lmindex)+np.abs(ind_localmax-rmindex))/2.*dt
            tim[k]   = ipar[0,imax]
            tvis[k]  = ipar[1,imax]
            ampgr[k] = ipar[2,imax]
            grvel[k] = dist/(tim[k] +tb) 
            snr[k]   = ipar[3,imax]
            wdth[k]  = ipar[4,imax] ### Note half width is not completely the same as ftanf77, need further check!!!
            phgr[k]  = ipar[5,imax]
            ipar_all.append(ipar)
        nfout1=nfin
        ################################################
        #       Check jumps in dispersion curve 
        ################################################
        grvel1=grvel.copy()
        tvis1 =tvis.copy()
        ampgr1=ampgr.copy()
        phgr1 =phgr.copy()
        snr1  =snr.copy()
        wdth1 =wdth.copy()
        ftrig1, trig1, ierr=self._trigger(grvel=grvel, om=omegaArr, tresh=tresh)
        # a=np.zeros(100)
        # b=np.zeros(100)
        # a[:64]=grvel
        # b[:64]=omegaArr
        # trig_old, ftrig_old, ierr=aftan.trigger(a, b, nfin, tresh)
        # return trig1, trig_old, ftrig1, ftrig_old
        if ierr !=0:
            deltrig1=trig1[1:]-trig1[:-1]
            ijmp=np.where(np.abs(deltrig1)>1.5)[0]
            if ijmp.size>1: ### need check
                delijmp=ijmp[1:]-ijmp[:-1]
                ii=np.where(delijmp<npoints)[0]
                ##########################################################################
                # Try to correct jumps by finding another local maximum as group arrival
                ##########################################################################
                if ii.size!=0:
                    grvelt=grvel1.copy()
                    tvist =tvis1.copy()
                    ampgrt=ampgr1.copy()
                    phgrt =phgr1.copy()
                    snrt  =snr1.copy()
                    wdtht =wdth1.copy()
                    for kk in ii:
                        istrt= ijmp[kk]
                        ibeg = istrt+1
                        iend = ijmp[kk+1]
                        ### nested loop, need further improvements
                        for i in xrange(iend-ibeg+1):
                            k=ibeg+i
                            ind_localmax=ind_all[k]
                            ipar=ipar_all[k]
                            wor = np.abs(dist/(ipar[0,:]+tb)-grvel1[k-1]) 
                            ima =wor.argmin() # find group arrivals that has smallest 
                            grvel1[k] = dist/(ipar[0,ima]+tb)
                            tvis1[k]  = ipar[1,ima]
                            ampgr1[k] = ipar[2,ima]
                            phgr1[k]  = ipar[5,ima]
                            snr1[k]   = ipar[3,ima]
                            wdth1[k]  = ipar[4,ima]
                        ftrig1, trig1, ierr=self._trigger(grvel=grvel1, om=omegaArr, tresh=tresh)
                        temptrig=trig1[istrt:iend+2]
                        if (np.where(temptrig>0.5)[0]).size==0:
                            grvelt=grvel1.copy()
                            tvist =tvis1.copy()
                            ampgrt=ampgr1.copy()
                            phgrt =phgr1.copy()
                            snrt  =snr1.copy()
                            wdtht =wdth1.copy()
                    grvel1=grvelt.copy()
                    tvis1 =tvist.copy()
                    ampgr1=ampgrt.copy()
                    phgr1 =phgrt.copy()
                    snr1  =snrt.copy()
                    wdth1 =wdtht.copy()
                    ftrig1, trig1, ierr=self._trigger(grvel=grvel1, om=omegaArr, tresh=tresh)
            ################################################
            # after correcting possible jumps, we cut frequency range to single
            # segment with maximum length
            ################################################
            if ierr!=0:
                indx=np.where(np.abs(trig1)>0.5)[0]
                indx=np.append(indx, nfin-1)
                iimax = indx[1:]-indx[:-1]
                ipos=iimax.argmax()
                ist = max(indx[ipos], 0)
                ibe = min(indx[ipos+1], nfin-1)
                nfout2= ibe-ist+1
                per1  =perArr[ist:ibe+1]
                grvel1=grvel1[ist:ibe+1]
                tvis1 =tvis1[ist:ibe+1]
                ampgr1=ampgr1[ist:ibe+1]
                phgr1 =phgr1[ist:ibe+1]
                snr1  =snr1[ist:ibe+1]
                wdth1 =wdth1[ist:ibe+1]
            else:
                nfout2 = nfin
                per1   = perArr.copy()
        else:
            nfout2 = nfin
            per1   = perArr.copy() 
        ################################################
        # fill out output data arrays
        ################################################
        if nprpv!=0:
            phV=self._phtovel(per=tvis, U=grvel, pha=phgr, npr=nprpv, prper=phprper[:nprpv], prvel=phprvel[:nprpv])
            phV1=self._phtovel(per=tvis1, U=grvel1, pha=phgr1, npr=nprpv, prper=phprper[:nprpv], prvel=phprvel[:nprpv])
        
        self.ftanparam.nfout1_1=nfin
        self.ftanparam.arr1_1=np.array([])
        self.ftanparam.nfout2_1=nfout2
        self.ftanparam.arr2_1=np.array([])
        self.ftanparam.tamp_1=
        self.ftanparam.nrow_1=0
        self.ftanparam.ncol_1=0
        self.ftanparam.ampo_1=np.array([],dtype='float32')
        self.ftanparam.ierr_1=0
        # Parameters for second iteration
        self.nfout1_2=0
        self.arr1_2=np.array([])
        self.nfout2_2=0
        self.arr2_2=np.array([])
        self.tamp_2=0.
        self.nrow_2=0
        self.ncol_2=0
        self.ampo_2=np.array([])
        self.ierr_2=0
        # Flag for existence of predicted phase dispersion curve
        self.preflag=False
        self.station_id=None
            
        #     
        #     U=np.append( grvel1, np.zeros(100-grvel1.size) )
        #     pha=np.append( phgr1, np.zeros(100-phgr1.size) )
        #     # # phtovel(delta,ip,n,per,U,pha,npr,prper,prvel,V)
        #     per=np.append( tvis1, np.zeros(100-tvis1.size) )
        #     phV_old=aftan.phtovel(dist, 1, tvis1.size, per, U, pha, nprpv, phprper, phprvel)
        # return phV, phV_old
        
        
        
        
        

        return
    
    def _compare_arr(self, data1, data2):
        plt.plot(data1-data2, '-y')
        plt.plot(data1, '^r')
        plt.plot(data2, '*b')
        plt.show()
        return
    
    def _compare_arr2(self, d1, d2, number):
        data1=d1[:, number]
        data2=d2[:data1.size, number]
        plt.plot(data1-data2, '-y')
        plt.plot(data1, '^r')
        plt.plot(data2, '*b')
        plt.show()
        return
    
    def taper(self, nb, ne, ntapb, ntape):
        omb = np.pi/ntapb
        ome = np.pi/ntape
        ncorr = int(ne+ntape)
        npts=self.stats.npts
        if ncorr>npts:
            ncorr=npts
        dataTapered=np.append(self.data[:ncorr], np.zeros( npts-ncorr ) )
        ##################################
        #zerp padding and cosine tapering
        ##################################
        # left end of the signal
        if nb-ntapb-1 > 0:
            dataTapered[:nb-ntapb-1]=0.
        if nb>ntapb:
            k=np.arange(ntapb+1)+nb-ntapb
            rwinb=(np.cos(omb*(nb-k))+1.)/2.
            dataTapered[nb-ntapb-1:nb]=rwinb*dataTapered[nb-ntapb-1:nb]
            sums = 2.*np.sum(rwinb)
        else:
            k=np.arange(nb)
            rwinb=(np.cos(omb*(nb-k))+1.)/2.
            dataTapered[:nb]=rwinb*dataTapered[:nb]
            sums = 2.*np.sum(rwinb)
        # right end of the signal
        if ne+ntape<npts:
            k=np.arange(ntape+1)+ne
            rwine=(np.cos(ome*(ne-k))+1.)/2.
            dataTapered[ne-1:ne+ntape] = dataTapered[ne-1:ne+ntape]*rwine
        elif ne < npts:
            k=np.arange(npts-ne+1)+ne
            rwine=(np.cos(ome*(ne-k))+1.)/2.
            dataTapered[ne-1:] = dataTapered[ne-1:]*rwine
        sums = sums+ne-nb-1
        c=np.sum(dataTapered[:ncorr])
        c=-c/sums
        # detrend
        if nb>ntapb:
            dataTapered[nb-ntapb-1:nb]=rwinb*c+dataTapered[nb-ntapb-1:nb]
        if ne+ntape<npts:
            dataTapered[ne-1:ne+ntape] = dataTapered[ne-1:ne+ntape] + rwine*c
        elif ne < npts:
            dataTapered[ne-1:] = dataTapered[ne-1:] + rwine*c
        dataTapered[nb:ne-1]=dataTapered[nb:ne-1]+c
        return dataTapered, ncorr
            
    
    # def _fmax(self, amp, pha, ind_j, ind_k, om, dt, t, dph, tm, ph, piover4 ):
    #     ind_jl=ind_j-1.
    #     ind_jr=ind_j+1.
    #     dd=amp[ind_jl, ind_k]+amp[ind_jr, ind_k]-2.*amp[ind_j, ind_k]
    #     t=(amp[ind_jl, ind_k]-amp[ind_jr, ind_k])/dd/2.0
    #     t[dd==0]=0.
    
    def _fmax(self, amp, pha, ind, om, piover4 ):
        dt=self.stats.delta
        ind_l=ind-1
        ind_r=ind+1
        dd=amp[ind_l]+amp[ind_r]-2.*amp[ind]
        dd[dd==0]=-9999
        t=(amp[ind_l]-amp[ind_r])/dd/2.0
        t[dd==-9999]=0.
        a1=pha[ind_l]
        a2=pha[ind]
        a3=pha[ind_r]
        k1 = (a2-a1-om*dt)/2./np.pi
        k1=np.round(k1)
        a2 = a2-2.*k1*np.pi
        k2 = (a3-a2-om*dt)/2./np.pi
        k2=np.round(k2)
        a3 = a3-2.*k2*np.pi
        dph=t*(a1+a3-2.*a2)+(a3-a1)/2.
        tm=t*t*(amp[ind_l]+amp[ind_r]-2.*amp[ind])/2.+t*(amp[ind_l]-amp[ind_r])/2.+amp[ind]
        ph=t*t*(a1+a3-2.*a2)/2.+t*(a3-a1)/2.+a2+np.pi*piover4/4.
        return dph, tm, ph, t
        
    def _trigger(self, grvel, om , tresh):
        nf=om.size
        hh1 = om[1:nf-1]-om[:nf-2]
        hh2 = om[2:]-om[1:nf-1]
        hh3 = hh1+hh2
        r = (grvel[:nf-2]/hh1 - (1./hh1+1/hh2)*grvel[1:nf-1] + grvel[2:]/hh2)*hh3/4.*100.
        ftrig=np.zeros(nf)
        ftrig[1:nf-1]=r
        trig=np.zeros(nf-2)
        trig[r>tresh]=1
        trig[r<-tresh]=-1
        trig=np.append(0, trig)
        trig=np.append(trig, 0)
        if (np.where(abs(trig)==1.)[0]).size!=0:
            ierr=1
        else:
            ierr=0
        return ftrig, trig, ierr
    
    def _phtovel(self, per, U, pha, npr, prper, prvel):
        dist=self.stats.sac.dist
        omegaArr=2.*np.pi/per
        T=dist/U
        sU=1./U
        # cs = CubicSpline(prper, prvel)
        # spl=scipy.interpolate.UnivariateSpline(prper, prvel, k=4, ext=3)
        spl=scipy.interpolate.CubicSpline(prper, prvel)
        # spl1=spl.derivative(1)
        # spl2=spl.derivative(2)
        Vpred=spl(per[-1])
        # ds=spl1(per[-1])
        # ds2=spl2(per[-1])
        phpred = omegaArr[-1]*(T[-1]-dist/Vpred)
        k=round((phpred -pha[-1])/2.0/np.pi)
        phV=np.zeros(U.size)
        phV[-1] = dist/(T[-1]-(pha[-1]+2.*k*np.pi)/omegaArr[-1])
        n=omegaArr.size
        for i in xrange(n-1):
            m=n-i-2
            Vpred =1/(((sU[m]+sU[m+1])*(omegaArr[m]-omegaArr[m+1])/2.+omegaArr[m+1]/phV[m+1])/omegaArr[m])
            phpred = omegaArr[m]*(T[m]-dist/Vpred)
            k = round((phpred -pha[m])/2.0/np.pi)
            phV[m] = dist/(T[m]-(pha[m]+2.0*k*np.pi)/omegaArr[m])
        return phV
        
    
    def aftanf77(self, pmf=True, piover4=-1.0, vmin=1.5, vmax=5.0, tmin=4.0, \
        tmax=30.0, tresh=20.0, ffact=1.0, taperl=1.0, snr=0.2, fmatch=1.0, phvelname='', predV=np.array([])):
        """ (Automatic Frequency-Time ANalysis) aftan analysis:
        ===========================================================================================================
        Input Parameters:
        pmf        - flag for Phase-Matched-Filtered output (default: True)
        piover4    - phase shift = pi/4*piover4, for cross-correlation piover4 should be -1.0
        vmin       - minimal group velocity, km/s
        vmax       - maximal group velocity, km/s
        tmin       - minimal period, s
        tmax       - maximal period, s
        tresh      - treshold for jump detection, usualy = 10, need modifications
        ffact      - factor to automatic filter parameter, usualy =1
        taperl     - factor for the left end seismogram tapering, taper = taperl*tmax,    (real*8)
        snr        - phase match filter parameter, spectra ratio to determine cutting point for phase matched filter
        fmatch     - factor to length of phase matching window
        phvelname  - predicted phase velocity file name
        predV      - predicted phase velocity curve, period = predV[:, 0],  Vph = predV[:, 1]
        
        Output:
        self.ftanparam, a object of ftanParam class, to store output aftan results
        ===========================================================================================================
        References:
        Levshin, A. L., and M. H. Ritzwoller. Automated detection, extraction, and measurement of regional surface waves.
             Monitoring the Comprehensive Nuclear-Test-Ban Treaty: Surface Waves. Birkh?user Basel, 2001. 1531-1545.
        Bensen, G. D., et al. Processing seismic ambient noise data to obtain reliable broad-band surface wave dispersion measurements.
             Geophysical Journal International 169.3 (2007): 1239-1260.
        """
        # preparing for data
        try:
            self.ftanparam
        except:
            self.init_ftanParam()
        try:
            dist=self.stats.sac.dist
        except:
            dist, az, baz=obspy.geodetics.base.gps2dist_azimuth(self.stats.sac.evla, self.stats.sac.evlo,
                                self.stats.sac.stla, self.stats.sac.stlo) # distance is in m
            self.stats.sac.dist=dist/1000.
            dist=dist/1000.
        nprpv = 0
        phprper=np.zeros(300)
        phprvel=np.zeros(300)
        if predV.size != 0:
            phprper=predV[:,0]
            phprvel=predV[:,1]
            nprpv = predV[:,0].size
            phprper=np.append( phprper, np.zeros(300-phprper.size) )
            phprvel=np.append( phprvel, np.zeros(300-phprvel.size) )
            self.ftanparam.preflag=True
        elif os.path.isfile(phvelname):
            # print 'Using prefile:',phvelname
            php=np.loadtxt(phvelname)
            phprper=php[:,0]
            phprvel=php[:,1]
            nprpv = php[:,0].size
            phprper=np.append( phprper, np.zeros(300-phprper.size) )
            phprvel=np.append( phprvel, np.zeros(300-phprvel.size) )
            self.ftanparam.preflag=True
        else:
            warnings.warn('No predicted dispersion curve for:'+self.stats.network+'.'+self.stats.station, UserWarning, stacklevel=1)
        nfin = 64
        npoints = 5  #  only 3 points in jump
        perc    = 50.0 # 50 % for output segment
        tempsac=self.copy()
        tb=self.stats.sac.b
        length=len(tempsac.data)
        if length>32768:
            warnings.warn('Length of seismogram is larger than 32768!', UserWarning, stacklevel=1)
            nsam=32768
            tempsac.data=tempsac.data[:nsam]
            tempsac.stats.e=(nsam-1)*tempsac.stats.delta+tb
            sig=tempsac.data
        else:
            sig=np.append(tempsac.data, np.zeros( 32768-tempsac.data.size, dtype='float64' ) )
            nsam=int( float (tempsac.stats.npts) )### for unknown reasons, this has to be done, nsam=int(tempsac.stats.npts)  won't work as an input for aftan
        dt=tempsac.stats.delta
        # Start to do aftan utilizing pyaftan
        self.ftanparam.nfout1_1,self.ftanparam.arr1_1,self.ftanparam.nfout2_1,self.ftanparam.arr2_1,self.ftanparam.tamp_1, \
                self.ftanparam.nrow_1,self.ftanparam.ncol_1,self.ftanparam.ampo_1, self.ftanparam.ierr_1= aftan.aftanpg(piover4, nsam, \
                    sig, tb, dt, dist, vmin, vmax, tmin, tmax, tresh, ffact, perc, npoints, taperl, nfin, snr, nprpv, phprper, phprvel)
        if pmf==True:
            if self.ftanparam.nfout2_1<3:
                return
            npred = self.ftanparam.nfout2_1
            tmin2 = self.ftanparam.arr2_1[1,0]
            tmax2 = self.ftanparam.arr2_1[1,self.ftanparam.nfout2_1-1]
            pred=np.zeros((2,300))
            pred[:,0:100]=self.ftanparam.arr2_1[1:3,:]
            pred=pred.T
            self.ftanparam.nfout1_2,self.ftanparam.arr1_2,self.ftanparam.nfout2_2,self.ftanparam.arr2_2,self.ftanparam.tamp_2, \
                    self.ftanparam.nrow_2,self.ftanparam.ncol_2,self.ftanparam.ampo_2, self.ftanparam.ierr_2 = aftan.aftanipg(piover4,nsam, \
                        sig,tb,dt,dist,vmin,vmax,tmin2,tmax2,tresh,ffact,perc,npoints,taperl,nfin,snr,fmatch,npred,pred,nprpv,phprper,phprvel)
        self.ftanparam.station_id=self.stats.network+'.'+self.stats.station 
        return

    def plotftan(self, plotflag=3, sacname=''):
        """
        Plot ftan diagram:
        This function plot ftan diagram.
        =================================================
        Input Parameters:
        plotflag -
            0: only Basic FTAN
            1: only Phase Matched Filtered FTAN
            2: both
            3: both in one figure
        sacname - sac file name than can be used as the title of the figure
        =================================================
        """
        try:
            fparam=self.ftanparam
            if fparam.nfout1_1==0:
                return "Error: No Basic FTAN parameters!"
            dt=self.stats.delta
            dist=self.stats.sac.dist
            if (plotflag!=1 and plotflag!=3):
                v1=dist/(fparam.tamp_1+np.arange(fparam.ncol_1)*dt)
                ampo_1=fparam.ampo_1[:fparam.ncol_1,:fparam.nrow_1]
                obper1_1=fparam.arr1_1[1,:fparam.nfout1_1]
                gvel1_1=fparam.arr1_1[2,:fparam.nfout1_1]
                phvel1_1=fparam.arr1_1[3,:fparam.nfout1_1]
                plb.figure()
                ax = plt.subplot()
                p=plt.pcolormesh(obper1_1, v1, ampo_1, cmap='gist_rainbow',shading='gouraud')
                ax.plot(obper1_1, gvel1_1, '--k', lw=3) #
                if (fparam.preflag==True):
                    ax.plot(obper1_1, phvel1_1, '--w', lw=3) #

                if (fparam.nfout2_1!=0):
                    obper2_1=fparam.arr2_1[1,:fparam.nfout2_1]
                    gvel2_1=fparam.arr2_1[2,:fparam.nfout2_1]
                    phvel2_1=fparam.arr2_1[3,:fparam.nfout2_1]
                    ax.plot(obper2_1, gvel2_1, '-k', lw=3) #
                    if (fparam.preflag==True):
                        ax.plot(obper2_1, phvel2_1, '-w', lw=3) #
                cb = plt.colorbar(p, ax=ax)
                Tmin1=obper1_1[0]
                Tmax1=obper1_1[fparam.nfout1_1-1]
                vmin1= v1[fparam.ncol_1-1]
                vmax1=v1[0]
                plt.axis([Tmin1, Tmax1, vmin1, vmax1])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('Basic FTAN Diagram '+sacname,fontsize=15)

            if fparam.nfout1_2==0 and plotflag!=0:
                return "Error: No PMF FTAN parameters!"
            if (plotflag!=0 and plotflag!=3):
                v2=dist/(fparam.tamp_2+np.arange(fparam.ncol_2)*dt)
                ampo_2=fparam.ampo_2[:fparam.ncol_2,:fparam.nrow_2]
                obper1_2=fparam.arr1_2[1,:fparam.nfout1_2]
                gvel1_2=fparam.arr1_2[2,:fparam.nfout1_2]
                phvel1_2=fparam.arr1_2[3,:fparam.nfout1_2]
                plb.figure()
                ax = plt.subplot()
                p=plt.pcolormesh(obper1_2, v2, ampo_2, cmap='gist_rainbow',shading='gouraud')
                ax.plot(obper1_2, gvel1_2, '--k', lw=3) #
                if (fparam.preflag==True):
                    ax.plot(obper1_2, phvel1_2, '--w', lw=3) #

                if (fparam.nfout2_2!=0):
                    obper2_2=fparam.arr2_2[1,:fparam.nfout2_2]
                    gvel2_2=fparam.arr2_2[2,:fparam.nfout2_2]
                    phvel2_2=fparam.arr2_2[3,:fparam.nfout2_2]
                    ax.plot(obper2_2, gvel2_2, '-k', lw=3) #
                    if (fparam.preflag==True):
                        ax.plot(obper2_2, phvel2_2, '-w', lw=3) #
                cb = plt.colorbar(p, ax=ax)
                Tmin2=obper1_2[0]
                Tmax2=obper1_2[fparam.nfout1_2-1]
                vmin2= v2[fparam.ncol_2-1]
                vmax2=v2[0]
                plt.axis([Tmin2, Tmax2, vmin2, vmax2])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('PMF FTAN Diagram '+sacname,fontsize=15)

            if ( plotflag==3 ):
                v1=dist/(fparam.tamp_1+np.arange(fparam.ncol_1)*dt)
                ampo_1=fparam.ampo_1[:fparam.ncol_1,:fparam.nrow_1]
                obper1_1=fparam.arr1_1[1,:fparam.nfout1_1]
                gvel1_1=fparam.arr1_1[2,:fparam.nfout1_1]
                phvel1_1=fparam.arr1_1[3,:fparam.nfout1_1]
                plb.figure(num=None, figsize=(18, 16), dpi=80, facecolor='w', edgecolor='k')
                ax = plt.subplot(2,1,1)
                p=plt.pcolormesh(obper1_1, v1, ampo_1, cmap='gist_rainbow',shading='gouraud')
                ax.plot(obper1_1, gvel1_1, '--k', lw=3) #
                if (fparam.preflag==True):
                    ax.plot(obper1_1, phvel1_1, '--w', lw=3) #
                if (fparam.nfout2_1!=0):
                    obper2_1=fparam.arr2_1[1,:fparam.nfout2_1]
                    gvel2_1=fparam.arr2_1[2,:fparam.nfout2_1]
                    phvel2_1=fparam.arr2_1[3,:fparam.nfout2_1]
                    ax.plot(obper2_1, gvel2_1, '-k', lw=3) #
                    if (fparam.preflag==True):
                        ax.plot(obper2_1, phvel2_1, '-w', lw=3) #
                cb = plt.colorbar(p, ax=ax)
                Tmin1=obper1_1[0]
                Tmax1=obper1_1[fparam.nfout1_1-1]
                vmin1= v1[fparam.ncol_1-1]
                vmax1=v1[0]
                plt.axis([Tmin1, Tmax1, vmin1, vmax1])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('Basic FTAN Diagram '+sacname)

                v2=dist/(fparam.tamp_2+np.arange(fparam.ncol_2)*dt)
                ampo_2=fparam.ampo_2[:fparam.ncol_2,:fparam.nrow_2]
                obper1_2=fparam.arr1_2[1,:fparam.nfout1_2]
                gvel1_2=fparam.arr1_2[2,:fparam.nfout1_2]
                phvel1_2=fparam.arr1_2[3,:fparam.nfout1_2]

                ax = plt.subplot(2,1,2)
                p=plt.pcolormesh(obper1_2, v2, ampo_2, cmap='gist_rainbow',shading='gouraud')
                ax.plot(obper1_2, gvel1_2, '--k', lw=3) #
                if (fparam.preflag==True):
                    ax.plot(obper1_2, phvel1_2, '--w', lw=3) #

                if (fparam.nfout2_2!=0):
                    obper2_2=fparam.arr2_2[1,:fparam.nfout2_2]
                    gvel2_2=fparam.arr2_2[2,:fparam.nfout2_2]
                    phvel2_2=fparam.arr2_2[3,:fparam.nfout2_2]
                    ax.plot(obper2_2, gvel2_2, '-k', lw=3) #
                    if (fparam.preflag==True):
                        ax.plot(obper2_2, phvel2_2, '-w', lw=3) #
                cb = plt.colorbar(p, ax=ax)
                Tmin2=obper1_2[0]
                Tmax2=obper1_2[fparam.nfout1_2-1]
                vmin2= v2[fparam.ncol_2-1]
                vmax2=v2[0]
                plt.axis([Tmin2, Tmax2, vmin2, vmax2])
                plt.xlabel('Period(s)')
                plt.ylabel('Velocity(km/s)')
                plt.title('PMF FTAN Diagram '+sacname)
        except AttributeError:
            print 'Error: FTAN Parameters are not available!'
        return