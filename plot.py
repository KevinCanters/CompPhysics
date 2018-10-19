# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:46:22 2017

@author: Lisa Bouckaert
"""

#!/usr/bin/env python3
"""Make a simple plot."""


 # ADD TO ROHF.PY OR UHF.PY
 #   with open("values.txt", "a") as myfile:
 #       chargeNa = 11- np.trace(np.dot(dm_alpha.T, S_A)) - np.trace(np.dot(dm_beta.T, S_A))
 #       ener = evals_alpha[nalpha-1] + evals_alpha[nalpha-2]+evals_alpha[nalpha-3] + evals_alpha[nalpha-4] + evals_alpha[nalpha-5]+ evals_beta[nbeta-1] + evals_beta[nbeta-2]+evals_beta[nbeta-3]
 #       myfile.write(str(ener) + "\t" + str(chargeNa) + "\n")
 #       myfile.close()
 
from glob import glob

import numpy as np
import matplotlib.pyplot as pt
import pylab

from uhf import load


def main():
    valence_energy, charge = np.loadtxt('valuesrohf.txt', unpack = 'true')
    valence_energy2, charge2 = np.loadtxt('valuesuhf.txt', unpack = 'true')
    table = []
    for fn_geoint in sorted(glob('geoint_*.npz')):
        geoint = load(fn_geoint)
        distance = np.linalg.norm(geoint['coordinates'][0] - geoint['coordinates'][1])
        energy_coulomb = 1/distance
        table.append([distance, energy_coulomb])
    table = np.array(table)
    convert = 0.0529177249
    
    pt.clf()
    pylab.plot((table[:,0] * convert), valence_energy + table[:,1], '-b', label='ROHF')
    pylab.plot((table[:,0] * convert), valence_energy2 + table[:,1], '-r', label='UHF')
    pylab.legend(loc='lower right')
    #pylab.ylim(-5.15 , -4.4)
    pt.ylabel('Energy [Hartree]')
    pt.xlabel('Internuclear distance [nm]')
    pt.savefig('plotrfcomp2.pdf')


if __name__ == '__main__':
    main()