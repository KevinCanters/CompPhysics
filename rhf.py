#!/usr/bin/env python3
"""Compute RHF energy for a given data file. Run as follows:

    ./rhf.py --help

or
    ./rhf.py fn_geoint fn_scfopt nalpha nbeta [fn_guess]
"""

import argparse

import numpy as np
from numpy.lib.npyio import _savez

from scipy.linalg import eigh


def main():
    """Main program."""
    args = parse_args()
    compute_hf(args.fn_geoint, args.fn_scfopt, args.nalpha, args.nbeta, args.fn_guess,
               args.nstep, args.threshold)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Parse command-line arguments.')
    parser.add_argument('fn_geoint', type=str, help='A geoint file.')
    parser.add_argument('fn_scfopt', type=str, help='The output file.')
    parser.add_argument('nalpha', type=int, help='Number of alpha electrons.')
    parser.add_argument('nbeta', type=int, help='Number of beta electrons.')    
    parser.add_argument('fn_guess', type=str, help='Initial guess for the SCF run.', nargs='?')
    parser.add_argument('-n', '--nstep', type=int, default=20, help='Max number of SCF iterations.')
    parser.add_argument('-t', '--threshold', type=float, default=1e-6, help='Convergence threshold.')
    return parser.parse_args()


def compute_hf(fn_geoint, fn_scfopt, nalpha, nbeta, fn_guess=None, nstep=20, threshold=1e-6):
    """Compute HF solution.
    
    Parameters
    ----------
    fn_geoint : str
        Name of file with geometry and integrals
    nalpha : int
        The number of alpha electrons (spin up).
    nbeta : int
        The number of beta electrons (spin down).
    fn_guess : str
        Optional file with previous solution to be used as guess.
    nstep : int
        The maximum number of SCF iterations.
    threshold : float
        The convergence threshold.
    """
    # Load geometry and integrals from the file
    geoint = load(fn_geoint)
    # Generate or load an initial guess of the density matrices.
    if fn_guess is None:
        guess = core_hamiltonian_guess(geoint, nalpha, nbeta)
    else:
        guess = load(fn_guess)
    # Optimize the wavefunction
    optimized = compute_scf(geoint, guess, nstep, threshold)
    # Write the solution to a file.
    dump(fn_scfopt, optimized)


def load(filename):
    """Load a dictionary of arrays from file."""
    return np.load(filename)


def dump(filename, data):
    """Dump a dictionary of arrays to file."""
    _savez(filename, [], data, True, allow_pickle=False)


def core_hamiltonian_guess(geoint, nalpha, nbeta):
    """Make an initial guess of the density matrices.
    
    Parameters
    ----------
    geoint : dict
        A dictionary with the geometry and integrals arrays
    nalpha : int
        The number of alpha electrons (spin up). Must be positive.
    nbeta : int
        The number of beta electrons (spin down). Must be positive.

    Returns
    -------
    guess : dict
        A dictionary with the initial guess density matrices, suitable for starting the
        SCF. It also contains nalpha, nbeta, and energy_nn (the nuclear-nuclear repulsion).
    """
    # Some sanity checks
    if nalpha < 0 or nbeta < 0 or nalpha + nbeta <= 0:
        raise ValueError('The number of alpha and beta electrons cannot be negative and '
                         'their sum should be strictly positive.')
    
    # Construct the initial guess
    guess = {'nalpha': nalpha, 'nbeta': nbeta}
    coreham = geoint['ki'] - geoint['nai']
    olp = geoint['oi']
    evals, evecs = eigh(coreham, olp)
    guess['evals_alpha'] = evals
    guess['evals_beta'] = evals
    guess['evecs_alpha'] = evecs
    guess['evecs_beta'] = evecs
    nalpha_rhf = (nalpha+nbeta)/2
    nbeta_rhf = nalpha_rhf
    guess['dm_alpha'] = np.dot(evecs[:, :nalpha_rhf], evecs[:, :nalpha_rhf].T)
    guess['dm_beta'] = np.dot(evecs[:, :nbeta_rhf], evecs[:, :nbeta_rhf].T)
    return guess
    

def compute_scf(geoint, guess, nstep=20, threshold=1e-6):
    """Basic RHF SCF solver.
    
    Parameters
    ----------
    geoint : dict
        A dictionary with the geometry and integrals arrays
    guess : dict
        A dictionary with nalpha, nbeta and initial guesses for dm_alpha and dm_beta
    nstep : int
        The maximum number of SCF iterations.
    threshold : float
        The convergence threshold.
    
    Returns
    -------
    optimized : dict
        A Dictionary with optimized dm_alpha, dm_beta, energy
    """
    # Some essential integrals...
    coreham = geoint['ki'] - geoint['nai']
    olp = geoint['oi']
    
    # The initial guess
    dm_alpha = guess['dm_alpha']
    dm_beta = guess['dm_beta']
    evals_alpha = None
    nalpha = guess['nalpha']
    nbeta = guess['nbeta']
    dm_total = dm_alpha + dm_beta

    # Repulsion between nuclei
    numbers = geoint['numbers']
    coordinates = geoint['coordinates']
    natom = len(numbers)
    energy_nn = 0.0
    for iatom0 in range(natom):
        for iatom1 in range(iatom0):
            distance = np.linalg.norm(coordinates[iatom0]-coordinates[iatom1])
            energy_nn += numbers[iatom0]*numbers[iatom1] / distance

    # The SCF cycle
    print('SCF Cycle')
    print('------------------------------------------------------')
    print('Iteration   Error alpha    Error beta           Energy')
    print('------------------------------------------------------')
    istep = 0
    while True:
        # Compute the Coulomb and exchange operators
        coulomb = np.tensordot(dm_total, geoint['eri'], ([0, 1], [0, 2]))
        exchange_alpha = np.tensordot(dm_alpha, geoint['eri'], ([0, 1], [0, 3]))
        exchange_beta = np.tensordot(dm_beta, geoint['eri'], ([0, 1], [0, 3]))

        # Compute the electronic energy
        ###START OF MODIFICATION (Making it correct for RHF)###
        energy = energy_nn+2*sum([
            expval(coreham, dm_total),
            expval(coulomb, dm_total),
            -0.5*expval(exchange_alpha, dm_alpha),
            -0.5*expval(exchange_beta, dm_beta),
        ])
        ###END OF MODIFICATION (Making it correct for RHF)###
        
        ###START OF MODIFICATION (Making it correct for RHF)###
        # Compute the Fock matrices
        fock_alpha = coreham + 2*coulomb - exchange_alpha
        fock_beta = coreham + 2*coulomb - exchange_beta
        ###END OF MODIFICATION (Making it correct for RHF)###

        # Compute convergence measures, using Pulay's commutator recipe.
        errors_alpha = np.dot(np.dot(fock_alpha, dm_alpha), olp) - \
                       np.dot(olp, np.dot(dm_alpha, fock_alpha))
        errors_beta = np.dot(np.dot(fock_beta, dm_beta), olp) - \
                      np.dot(olp, np.dot(dm_beta, fock_beta))
        error_alpha = np.sqrt((errors_alpha**2).mean())
        error_beta = np.sqrt((errors_beta**2).mean())

        # Print some stuff and break if converged or when maximum number of iterations
        # has been reached.
        istep += 1
        print('{:9d}  {:10.6e}  {:10.6e}  {:15.6f}'.format(
            istep, error_alpha, error_beta, energy))
        if (error_alpha < threshold and error_beta < threshold):
            print('Converged')
            converged = True
            break
        elif istep > nstep:
            print('Not converged')
            converged = False
            break

        # Construct new occupied and virtual orbitals
        evals_alpha, evecs_alpha = eigh(fock_alpha, olp)
        evals_beta, evecs_beta = eigh(fock_beta, olp)

        # Construct the density matrices
        ###START OF MODIFICATION (Making nalpha = nbeta = N/2)###
        nalpha_rhf = (nalpha+nbeta)/2
        nbeta_rhf = nalpha_rhf
        dm_alpha = np.dot(evecs_alpha[:, :nalpha_rhf], evecs_alpha[:, :nalpha_rhf].T)
        dm_beta = np.dot(evecs_beta[:, :nbeta_rhf], evecs_beta[:, :nbeta_rhf].T)
        dm_total = dm_alpha + dm_beta
        ###END OF MODIFICATION (Making nalpha = nbeta = N/2)###
    print('------------------------------------------------------')

    # print some things on screen
    ###START OF MODIFICATION (Making it correct for RHF (Na = Nb)###
    nalpha_rhf = (nalpha+nbeta)/2
    nbeta_rhf = nalpha_rhf
    ###END OF MODIFICATION (Making it correct for RHF (Na = Nb)###
    print('Energy [Hartree]:', energy)
    if evals_alpha is None:
        print('No orbital energies were computed.')
    else:
        print('Orbital energies [Hartree]:')
        for iorb in range(len(olp)):
            print('    {:1s} {:10.5f}     {:1s} {:10.5f}'.format(
                [' ', 'a'][int(iorb<nalpha)],
                evals_alpha[iorb],
                [' ', 'b'][int(iorb<nbeta)],
                evals_beta[iorb],
            ))
    
    ###START OF MODIFICATION (Adding checks for the new S_A and S_B functions + Checking computations with them)###
    print('------------------------------------------------------')
    print('Checking overlap equations and electrons on atoms')
    print('------------------------------------------------------')   
    # Create overlap matrix S_A and S_B
    S_A=np.zeros(shape=(34,34))
    S_B=np.zeros(shape=(34,34))
    for i in range(34):
        for j in range(34):
            if(i < 17):
                if(j < 17):
                    S_A[i][j] = 1*olp[i][j]
                    S_B[i][j] = 0
                else:
                    S_A[i][j] = 0.5*olp[i][j]
                    S_B[i][j] = 0.5*olp[i][j]
            else:
                if(j < 17):
                    S_A[i][j] = 0.5*olp[i][j]
                    S_B[i][j] = 0.5*olp[i][j]
                else:
                    S_A[i][j] = 0
                    S_B[i][j] = 1*olp[i][j]
                    
    # Check if S_A + S_B = S 
    if np.array_equal(olp,(S_A + S_B)): print('Overlap matrices combine to S! (S = S_A + S_B)')  
    else: print('Overlap matrices don\'t overlap!')
    
    # Check total number of alpha and beta electrons Tr(D.T * S)
    print('Number of alpha electrons: ' + str(np.trace(np.dot(dm_alpha.T, olp))))
    print('Number of beta electrons: ' + str(np.trace(np.dot(dm_beta.T, olp))))
    
    # Control alpha/beta on A/B
    print ('Alpha on A:\t' + str(np.trace(np.dot(dm_alpha.T, S_A))))
    print ('Beta on A:\t' + str(np.trace(np.dot(dm_beta.T, S_A))))
    print ('Alpha on B:\t' + str(np.trace(np.dot(dm_alpha.T, S_B))))
    print ('Beta on B:\t' + str(np.trace(np.dot(dm_beta.T, S_B))))
    ###END OF MODIFICATION (Adding checks for the new S_A and S_B functions + Checking computations with them)###
    
    # Construct the return dictionary
    return {
        'nalpha': nalpha,
        'nbeta': nbeta,
        'dm_alpha': dm_alpha,
        'dm_beta': dm_beta,
        'dm_total': dm_total,
        'energy': energy,
        'energy_nn': energy_nn,
        'converged': converged,
        'error_alpha': error_alpha,
        'error_beta': error_beta,
        'istep': istep,
    }


def expval(op, dm):
    """Compute the expectation value for a given operator and density matrix."""
    return np.tensordot(op, dm, ([0, 1], [0, 1]))


if __name__ == '__main__':
    main()
