import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import polymer
import matplotlib.pyplot as plt
import scipy
import statsmodels.api as sm

#from cmeutils.sampling import autocorr1D

def autocorr1D(array):
    """Takes in a linear np array, performs autocorrelation
    function and returns normalized array with half the length
    of the input.

    Parameters
    ----------
    data : numpy.typing.Arraylike, required
        1-D series of data to perform autocorrelation on.

    Returns
    -------
    1D np.array

    """
    ft = np.fft.rfft(array - np.average(array))
    acorr = np.fft.irfft(ft * np.conjugate(ft)) / (len(array) * np.var(array))
    return acorr[0 : len(acorr) // 2]  # noqa: E203

def get_decorr(acorr):
    """
    Returns the decorrelation time of the autocorrelation, a 1D np array
    """
    return np.argmin(acorr > 0)

def persistence_length(filepath,start,stop,interval):
    """
    filepath needs to be a format in which you can
    create an mdanalysis universe from, we mostly use gsd files
    """
    u = mda.Universe(topology=filepath)

    """rewrite atom indices"""
    bond_indices = []
    particle_index = [0]
    for i in range(len(u.bonds)):
        a = u.bonds[i].atoms.indices
        bond_indices.append(list(a))
        if particle_index[-1] in bond_indices[i]:
            atom1 = bond_indices[i][0]
            atom2 = bond_indices[i][1]
            if atom1 not in particle_index:
                particle_index.append(atom1)
            if atom2 not in particle_index:
                particle_index.append(atom2)
 #   print(bond_indices)

    """create bonds list"""
    autocorrelation = []
    bond_len = []
    for t in u.trajectory[start:stop:interval]:
        particle_positions = []
        bonds = []
        unit_bonds = []
        bond_lengths = []
        angles = []

        for i in particle_index:
            pos = t.positions[i]
            particle_positions.append(pos)
        for i in range(len(u.bonds)):
            b = particle_positions[i+1]-particle_positions[i]
            bonds.append(b)
            l2 = t.dimensions[0]/2
            for i,b in enumerate(bonds):
                for j,x in enumerate(b):
                    if x>l2:
                        bonds[i][j] = x-l2*2
                    if x<-l2:
                        bonds[i][j] = x+l2*2
            a = b/np.linalg.norm(b)
            unit_bonds.append(a)
            length = np.linalg.norm(b)
            bond_lengths.append(length)
            #l_b = np.mean(bond_lengths)
        bond_len.append(bond_lengths)

#        angles.append(np.dot(unit_bonds,unit_bonds))

        for i in range(len(unit_bonds)-1):
            b1 = unit_bonds[0]
            b2 = unit_bonds[0+i]
            dot_product = np.dot(b1,b2)
            angles.append(dot_product)

        n=len(u.bonds)
        n_frames = u.trajectory.n_frames
        n_chains = 1
        norm = np.linspace(1,n- 1, n - 1)
        norm *= n_chains# * n_frames
        #auto = autocorr1D(angles)
        #auto = sm.tsa.acf(angles)
        #autocorrelation.append(auto)
        autocorrelation.append(angles)#/norm)
#        print(angles) 

    '''average the data from trajectories together'''
    auto_average = []
    for j in range(len(autocorrelation[0])):
        k = []
        for i in range(len(autocorrelation)):
            k.append(autocorrelation[i][j])
        auto_average.append(np.mean(k))
#    print(auto_average)

    l_b = np.average(bond_len)
#    result = [x/len(av) for x in sums]
    x = [i for i in range(len(auto_average))]

    '''set negative results to 0'''
    for r in range(len(auto_average)):
        if auto_average[r] < 0:
            auto_average[r] = 0
    def expfunc(x, a):
        return np.exp(-x/a)

    exp_coeff = scipy.optimize.curve_fit(expfunc,x,auto_average)[0][0]

    l_p = exp_coeff * l_b

    fit = np.exp(-(x/exp_coeff))
    decorrelation = get_decorr(np.array(auto_average))

    return l_p, l_b, x, auto_average, fit, exp_coeff, autocorrelation, decorrelation, unit_bonds
