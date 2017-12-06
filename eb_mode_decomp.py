#/usr/bin/env python
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

class SimsTQU():
    """
    Class for TQU maps from Blakesley's sims
    Can specify either TQU=[T, Q, U] or a filename
    """
    def __init__(self, TQU= None, fn=None):
        # Specify maps by hand
        if TQU is not None:
            self.T = T
            self.Q = Q
            self.U = U
            
        # Or: Load from filename
        if fn is not None:
            #root = "/Users/burkhart1/Desktop/EEBB/qu_data/"
            root = "/Users/susanclark/Dropbox/ModeDecomp/BlakesleyModes/AllMode/"
            self.T = fits.getdata(root+"Imap_"+fn+".fits")
            self.Q = fits.getdata(root+"q_"+fn+".fits")
            self.U = fits.getdata(root+"u_"+fn+".fits")
            
        self.fn = fn
        
        # shape parameters
        self.ny, self.nx = self.T.shape
        self.midpoint = np.int(self.nx/2)
        
        # should be a square simulation box
        assert(self.ny == self.nx)
        
        self.boxtheta = 0.008  # image size (radians) ... where did this come from?
        self.dx = self.boxtheta/self.nx
        self.dy = self.boxtheta/self.ny

        # Check that maps are all the same shape
        assert( (self.ny, self.nx) == self.T.shape )
        assert( (self.ny, self.nx) == self.Q.shape )
        assert( (self.ny, self.nx) == self.U.shape )
        
        

    def get_tebfft(self):
        """
        taken from quicklens
        """
        
        #(lx, ly) pair associated with each Fourier mode in T, E, B.
        (self.lx, self.ly) = np.meshgrid( np.fft.fftfreq( self.nx, self.dx )[0:self.midpoint+1]*2.*np.pi,
                    np.fft.fftfreq( self.ny, self.dy )*2.*np.pi )
                    
        # Eqns from Kovetz & Kamionkowski
        tpi  = 2.*np.arctan2(self.lx, -self.ly)
        tfac = np.sqrt((self.dx * self.dy) / (self.nx * self.ny))
        qfft = np.fft.rfft2(self.Q) * tfac
        ufft = np.fft.rfft2(self.U) * tfac
        
        self.T_fft = np.fft.rfft2(self.T) * tfac
        self.E_fft = (+np.cos(tpi) * qfft + np.sin(tpi) * ufft)
        self.B_fft = (-np.sin(tpi) * qfft + np.cos(tpi) * ufft)
        
    def teb_map(self):
        """
        map T, E, B back into image space
        """
        
        self.T_map = np.fft.irfft2(self.T_fft)
        self.E_map = np.fft.irfft2(self.E_fft)
        self.B_map = np.fft.irfft2(self.B_fft)
        
    def get_tt_ee_bb(self):
        """
        get auto power spectra. hacked together from quicklens
        """
        
        # for l-bin averaging
        self.lmin = 2.*np.pi/self.boxtheta
        self.lmax = 2.*np.pi/self.dx
        self.lbins = np.linspace(self.lmin, self.lmax, self.nx/2)
        
        self.psimin = 0
        self.psimax = np.inf
        self.psispin = 1

        # all ells
        self.ell = np.sqrt(self.lx**2 + self.ly**2).flatten()

        wvec = np.ones(self.ell.shape)
        C_ell = lambda l:l*(l+1.)/(2.*np.pi)
        tvec = C_ell(self.ell)

        norm, bins = np.histogram(self.ell, bins=self.lbins, weights=wvec); norm[ np.where(norm != 0.0) ] = 1./norm[ np.where(norm != 0.0) ]
        self.cltt, bins = np.histogram(self.ell, bins=self.lbins, weights=wvec*tvec*(self.T_fft * np.conj(self.T_fft)).flatten().real); self.cltt *= norm
        self.clte, bins = np.histogram(self.ell, bins=self.lbins, weights=wvec*tvec*(self.T_fft * np.conj(self.E_fft)).flatten().real); self.clte *= norm
        self.cltb, bins = np.histogram(self.ell, bins=self.lbins, weights=wvec*tvec*(self.T_fft * np.conj(self.B_fft)).flatten().real); self.cltb *= norm
        self.clee, bins = np.histogram(self.ell, bins=self.lbins, weights=wvec*tvec*(self.E_fft * np.conj(self.E_fft)).flatten().real); self.clee *= norm
        self.cleb, bins = np.histogram(self.ell, bins=self.lbins, weights=wvec*tvec*(self.E_fft * np.conj(self.B_fft)).flatten().real); self.cleb *= norm
        self.clbb, bins = np.histogram(self.ell, bins=self.lbins, weights=wvec*tvec*(self.B_fft * np.conj(self.B_fft)).flatten().real); self.clbb *= norm
        
        # bin centers
        self.ls = 0.5*(self.lbins[0:-1] + self.lbins[1:])
        
    def plot_allmaps(self):
        """
        make a plot of T, Q, U and T, E, B
        """
        
        fig = plt.figure()
        alltitles = ["T", "Q", "U", "T", "E", "B"]
        for _i, map in enumerate([self.T, self.Q, self.U, self.T_map, self.E_map, self.B_map]):
            ax = fig.add_subplot(2, 3, _i + 1)
            ax.imshow(map)
            ax.set_title(alltitles[_i])
        
        plt.suptitle(self.fn)
        
    def plot_all_autopower_spectra(self):
        """
        make a plot of TT, EE, BB
        """
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for _i, cc in enumerate([self.cltt, self.clee, self.clbb]):
            ax.loglog(self.ls, cc)
        
        plt.legend(['TT', 'EE', 'BB'])
        plt.xlabel('ell')
        plt.ylabel(r'$C_l$')
        
    def plot_TE(self):
        """
        make a plot of TE
        """
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for _i, cc in enumerate([self.clte]):
            ax.loglog(self.ls, cc)
        
        plt.legend(['TE'])
        plt.xlabel('ell')
        plt.ylabel(r'$C_l$')
        
        
testTQU = SimsTQU(fn="c512b.1p.01_z")
testTQU.get_tebfft()
testTQU.teb_map()
#testTQU.plot_allmaps()
testTQU.get_tt_ee_bb()
#testTQU.plot_all_autopower_spectra()
testTQU.plot_TE()




