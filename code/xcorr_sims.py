import numpy as np
import pymaster as nmt
import h5py
import time
from astropy.io import fits

class SimsTQU():
    """
    Class for TQU maps from Blakesley's sims
    Can specify either TQU=[T, Q, U] or a filename
    """
    def __init__(self, TQU= None, fn=None, modedecomp=False, ikey=""):
        # Specify maps by hand
        if TQU is not None:
            self.T = T
            self.Q = Q
            self.U = U
            
        # Or: Load from filename
        if fn is not None:
            if modedecomp:
                root = "/data/seclark/BlakesleySims/simdata/ModeDecomp/"
                self.T = fits.getdata(root+"Imap_"+ikey+".fits")
                self.Q = fits.getdata(root+"q_"+fn+".fits")
                self.U = fits.getdata(root+"u_"+fn+".fits")
            else:
                root = "/data/seclark/BlakesleySims/simdata/"
                self.T = fits.getdata(root+"Imap_"+fn+".fits")
                self.Q = fits.getdata(root+"q_"+fn+".fits")
                self.U = fits.getdata(root+"u_"+fn+".fits")
                
        self.Q[np.where(np.isnan(self.Q) == True)] = 0
        self.U[np.where(np.isnan(self.U) == True)] = 0
            
        self.fn = fn
        
        # shape parameters
        self.ny, self.nx = self.T.shape
        self.midpoint = np.int(self.nx/2)
        
        # should be a square simulation box
        assert(self.ny == self.nx)
        
        #self.boxtheta = 0.008  # image size (radians) ... where did this come from?
        #self.dx = self.boxtheta/self.nx
        #self.dy = self.boxtheta/self.ny

        # Check that maps are all the same shape
        assert( (self.ny, self.nx) == self.T.shape )
        assert( (self.ny, self.nx) == self.Q.shape )
        assert( (self.ny, self.nx) == self.U.shape )
        
    
def xcorr_flatsky(modedecomp=False, simkey="512_alfven3_0002_a_z", Imapkey="", deglen=10, apotype="C2", aposcale=0.5, Epure=True, Bpure=True):
    
    TQU = SimsTQU(fn=simkey, modedecomp=modedecomp, ikey=Imapkey)

    # Define flat-sky field
    #  - Lx and Ly: the size of the patch in the x and y dimensions (in radians)
    Lx = deglen * np.pi/180. # arbitrarily set this to a deglen x deglen deg box
    Ly = deglen * np.pi/180.
    #  - Nx and Ny: the number of pixels in the x and y dimensions
    Nx = 512
    Ny = 512

    # Define mask
    mask=np.ones_like(TQU.T).flatten()
    xarr=np.ones(Ny)[:,None]*np.arange(Nx)[None,:]*Lx/Nx
    yarr=np.ones(Nx)[None,:]*np.arange(Ny)[:,None]*Ly/Ny
    #Let's also trim the edges
    mask[np.where(xarr.flatten()<Lx/16.)]=0; mask[np.where(xarr.flatten()>15*Lx/16.)]=0;
    mask[np.where(yarr.flatten()<Ly/16.)]=0; mask[np.where(yarr.flatten()>15*Ly/16.)]=0;
    mask=mask.reshape([Ny,Nx])
    mask = nmt.mask_apodization_flat(mask, Lx, Ly, aposize=aposcale, apotype=apotype)

    # Fields:
    # Once you have maps it's time to create pymaster fields.
    # Note that, as in the full-sky case, you can also pass
    # contaminant templates and flags for E and B purification
    # (see the documentation for more details)
    f0 = nmt.NmtFieldFlat(Lx, Ly, mask, [TQU.T])
    f2 = nmt.NmtFieldFlat(Lx, Ly, mask, [TQU.Q, TQU.U], purify_b=Bpure, purify_e=Epure)

    # Bins:
    # For flat-sky fields, bandpowers are simply defined as intervals in ell, and
    # pymaster doesn't currently support any weighting scheme within each interval.
    l0_bins = np.arange(Nx/8) * 8 * np.pi/Lx
    lf_bins = (np.arange(Nx/8)+1) * 8 * np.pi/Lx
    b = nmt.NmtBinFlat(l0_bins, lf_bins)
    # The effective sampling rate for these bandpowers can be obtained calling:
    ells_uncoupled = b.get_effective_ells()
    
    # Workspaces:
    # As in the full-sky case, the computation of the coupling matrix and of
    # the pseudo-CL estimator is mediated by a WorkspaceFlat case, initialized
    # by calling its compute_coupling_matrix method:
    w00 = nmt.NmtWorkspaceFlat()
    w00.compute_coupling_matrix(f0, f0, b)
    w02 = nmt.NmtWorkspaceFlat()
    w02.compute_coupling_matrix(f0, f2, b)
    w22 = nmt.NmtWorkspaceFlat()
    w22.compute_coupling_matrix(f2, f2, b)
    
    # Computing power spectra:
    # As in the full-sky case, you compute the pseudo-CL estimator by
    # computing the coupled power spectra and then decoupling them by
    # inverting the mode-coupling matrix. This is done in two steps below,
    # but pymaster provides convenience routines to do this
    # through a single function call
    cl00_coupled = nmt.compute_coupled_cell_flat(f0, f0, b)
    cl00_uncoupled = w00.decouple_cell(cl00_coupled)
    cl02_coupled = nmt.compute_coupled_cell_flat(f0, f2, b)
    cl02_uncoupled = w02.decouple_cell(cl02_coupled)
    cl22_coupled = nmt.compute_coupled_cell_flat(f2, f2, b)
    cl22_uncoupled = w22.decouple_cell(cl22_coupled)
    
    TT = cl00_uncoupled[0]
    TE = cl02_uncoupled[0]
    TB = cl02_uncoupled[1]
    EE = cl22_uncoupled[0]
    EB = cl22_uncoupled[1]
    BE = cl22_uncoupled[2]
    BB = cl22_uncoupled[3]
    
    if modedecomp:
        outroot = "/data/seclark/BlakesleySims/simdata/ModeDecomp/xcorrdata/"
    else:
        outroot = "/data/seclark/BlakesleySims/xcorrdata/"
    outfn = simkey+"_deglen{}_{}apod{}_EBpure{}{}.h5".format(deglen, apotype, aposcale, Epure, Bpure)
    
    with h5py.File(outroot+outfn, 'w') as f:
        
        TTdset = f.create_dataset(name='TT', data=TT)
        TEdset = f.create_dataset(name='TE', data=TE)
        TBdset = f.create_dataset(name='TB', data=TB)
        EEdset = f.create_dataset(name='EE', data=EE)
        EBdset = f.create_dataset(name='EB', data=EB)
        BEdset = f.create_dataset(name='BE', data=BE)
        BBdset = f.create_dataset(name='BB', data=BB)
        TTdset.attrs['deglen'] = deglen
        TTdset.attrs['Epure'] = Epure
        TTdset.attrs['Bpure'] = Bpure
        TTdset.attrs['ell_binned'] = ells_uncoupled
        
if __name__ == "__main__":
    
    apotype = "C2"
    deglen = 10
    aposcale = 2.0 #0.5
    Epure = True
    Bpure = True
    modedecomp = False
    
    if modedecomp:
        #allsimkeys = ["b{}p{}_{}_z".format(b, p, wave) for b in [".1", ".5", "1", "3", "5"] for p in [".01", "2"] for wave in ["alf", "fast", "slow"]]
        #allImapkeys = ["c512b{}p{}_z".format(b, p, wave) for b in [".1", ".5", "1", "3", "5"] for p in [".01", "2"] for wave in ["alf", "fast", "slow"]]
        allsimkeys = ["c512b{}p{}_{}".format(b, p, d) for b in [".1", ".5", "1", "3", "5"] for p in [".01", "2"] for d in ["x", "y", "z"]]
        allImapkeys = ["c512b{}p{}_{}".format(b, p, d) for b in [".1", ".5", "1", "3", "5"] for p in [".01", "2"] for d in ["x", "y", "z"]]
    else:
        # all sim keys for non-mode decomp sims
        allsimkeys = ["512_alfven{}_000{}_{}_x".format(i, n, ae) for n in np.arange(1, 5) for i in [1, 3, 6] for ae in ["a", "e"]]

    
    if modedecomp:
        for _i, (skey, ikey) in enumerate(zip(allsimkeys, allImapkeys)):
            time0 = time.time()
            try:
                xcorr_flatsky(modedecomp=modedecomp, simkey=skey, Imapkey=ikey, deglen=deglen, apotype=apotype, aposcale=aposcale, Epure=Epure, Bpure=Bpure)
            except IOError:
                print("{} does not exist".format(skey))  
            time1 = time.time()
              
            print("skey {} took {} minutes".format(_i, (time1 - time0)/60.))
    
    else:
        for _i, skey in enumerate(allsimkeys):
            time0 = time.time()
            xcorr_flatsky(modedecomp=modedecomp, simkey=skey, deglen=deglen, apotype=apotype, aposcale=aposcale, Epure=Epure, Bpure=Bpure)  
            time1 = time.time()
              
            print("skey {} took {} minutes".format(_i, (time1 - time0)/60.))
        
    