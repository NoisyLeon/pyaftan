c======================================================================
c aftanipg function. Provides ftan analysis with phase match filter,
c jumps correction, phase velocity computation and amplitude map 
c output for input periods.
c
c
c Autor: M. Barmine,CIEI,CU. Date: Jun 15, 2006. Version: 2.00
c
      subroutine aftanipg(piover4,n,sei,t0,dt,delta,vmin,vmax,tmin,tmax,
     *           tresh,ffact,perc,npoints,taperl,nfin,fsnr,fmatch,npred,
     *           pred,nphpr,phprper,phprvel,nfout1,arr1,nfout2,arr2,
     *           tamp,nrow,ncol,amp,ierr)
c======================================================================
c Parameters for aftanipg function:
c Input parameters:
c piover4 - phase shift = pi/4*piover4, for cross-correlation
c           piover4 should be   -1.0 !!!!     (real*8)
c n       - number of input samples, (integer*4)
c sei     - input array length of n, (real*4)
c t0      - time shift of SAC file in seconds, (real*8)
c dt      - sampling rate in seconds, (real*8)
c delta   - distance, km (real*8)
c vmin    - minimal group velocity, km/s (real*8)
c vmax    - maximal value of the group velocity, km/s (real*8)
c tmin    - minimal period, s (real*8)
c tmax    - maximal period, s (real*8)
c tresh   - treshold, usualy = 10, (real*8)
c ffact   - factor to automatic filter parameter, usualy =1, (real*8)
c perc    - minimal length of of output segment vs freq. range, % (real*8)
c npoints - max number points in jump, (integer*4)
c taperl  - factor for the left end seismogram tapering,
c           taper = taperl*tmax,    (real*8)
c nfin    - starting number of frequencies, nfin <= 100,(integer*4)
c fsnr    - phase match filter parameter, spectra ratio to 
c           determine cutting point   (real*8)
c fmatch  - factor to length of phase matching window (real*8)
c npred   - length of the group velocity prediction table
c pred    - group velocity prediction table:    (real*8)
c pred(0,:) - periods of group velocity prediction table, s
c pred(1,:) - pedicted group velocity, km/s
c nphpr   - length of phprper and phprvel arrays
c phprper - predicted phase velocity periods, s
c phprvel - predicted phase velocity for corresponding periods, s
c ==========================================================
c Output parameters are placed in 2-D arrays arr1 and arr2,
c arr1 contains preliminary results and arr2 - final.
c ==========================================================
c nfout1 - output number of frequencies for arr1, (integer*4)
c arr1   - preliminary results.
c          Description: real*8 arr1(8,n), n >= nfin)
c          arr1(1,:) -  central periods, s (real*8)
c          arr1(2,:) -  apparent periods, s (real*8)
c          arr1(3,:) -  group velocities, km/s (real*8)
c          arr1(4,:) -  phase velocities, km/s (real*8)
c          arr1(5,:) -  amplitudes, Db (real*8)
c          arr1(6,:) -  discrimination function, (real*8)
c          arr1(7,:) -  signal/noise ratio, Db (real*8)
c          arr1(8,:) -  maximum half width, s (real*8)
c arr2   - final results
c nfout2 - output number of frequencies for arr2, (integer*4)
c          Description: real*8 arr2(7,n), n >= nfin)
c          If nfout2 == 0, no final results.
c          arr2(1,:) -  central periods, s (real*8)
c          arr2(2,:) -  apparent periods, s (real*8)
c          arr2(3,:) -  group velocities, km/s (real*8)
c          arr1(4,:) -  phase velocities, km/s (real*8)
c          arr2(5,:) -  amplitudes, Db (real*8)
c          arr2(6,:) -  signal/noise ratio, Db (real*8)
c          arr2(7,:) -  maximum half width, s (real*8)
c          tamp      -  time to the beginning of ampo table, s (real*8)
c          nrow      -  number of rows in array ampo, (integer*4)
c          ncol      -  number of columns in array ampo, (integer*4)
c          amp       -  Ftan amplitude array, Db, (real*8)
c ierr   - completion status, =0 - O.K.,           (integer*4)
c                             =1 - some problems occures
c                             =2 - no final results
c======================================================================
      implicit none
      integer*4 n,npoints,nf,nfin,nfout1,ierr,nrow,ncol
      real*8    piover4,perc,taperl,tamp,arr1(8,100),arr2(7,100)
      real*8    t0,dt,delta,vmin,vmax,tmin,tmax,tresh,ffact,ftrig(100)
      real*8    fsnr,fmatch
      real*4    sei(32768)
      double complex dczero,s(32768),sf(32768),fils(32768),tmp(32768)
      real*8    grvel(100),tvis(100),ampgr(100),om(100),per(100)
      real*8    tim(100)
