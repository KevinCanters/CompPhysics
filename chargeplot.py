# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:41:34 2017

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
    valence_energy, charge = np.loadtxt('values.txt', unpack = 'true')
    print ( len(charge))
    table = []
    for fn_geoint in sorted(glob('geoint_*.npz')):
        geoint = load(fn_geoint)
        distance = np.linalg.norm(geoint['coordinates'][0] - geoint['coordinates'][1])
        energy_coulomb = 1/distance
        table.append([distance, energy_coulomb])
    table = np.array(table)
    converse = 0.0529177249
    
    pt.clf()
    pylab.plot((table[:,0] * converse), charge, '-b', label='UHF')
    pt.ylabel('Charge of Natrium')
    pt.xlabel('Internuclear distance [nm]')
    pt.savefig('chargeplotrf.pdf')


if __name__ == '__main__':
    main()