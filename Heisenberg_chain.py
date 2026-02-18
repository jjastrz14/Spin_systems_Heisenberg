# version 3.0 for spin 1D chain diagonalization (max number of sites 8-10) using numpy and:
# -> calculating a density matrix
# -> calculating a reduced density matrix of chain divided into two equal subsytems
# -> calculating entropy of this subsystems
# -> making a matrix block diagonal by simple matrix operations
# -> calculation of rho reduced density matrices with fixed s_z quantum number
####

import numpy as np
from functools import reduce
from itertools import chain, product
import os
import math
import scipy.sparse

ZERO_THRESHOLD  = 1e-8    # eigenvalues of rho below this treated as zero
TRACE_TOLERANCE = 1e-12   # tolerance for Tr(rho) == 1

class Heisenberg(object):

    #Creating Heisenberg Hamiltonian

    #Initialization of the system and spin opearators
    def __init__(self,N, S, directory = None) -> None:
        self.size_of_system = N
        self.chain_I = []
        self.S_site_whole = []
        self.energies = []
        self.vectors = []
        self.possible_basis = []
        self.H = 0
        self.S = S
        self.basis = []
        self.list_of_spins = []
        self.list_spins = []
        self.basis_s_z = []

        assert S in (1/2, 1), f"Unsupported spin value S={S}. Only S=1/2 and S=1 are implemented."

        # To generalize to arbitrary S, use ladder operator formulas:
        #   S_plus |S,m> = sqrt(S(S+1) - m(m+1)) |S,m+1>
        #   S_minus|S,m> = sqrt(S(S+1) - m(m-1)) |S,m-1>
        # to build (2S+1) x (2S+1) matrices for any S.

        if S == 1:
        #matrices for S = 1
            self.S_plus = np.sqrt(2) * np.array([[0,1,0],
                                            [0,0,1],
                                            [0,0,0]])

            self.S_minus = np.sqrt(2) * np.array([[0,0,0],
                                            [1,0,0],
                                            [0,1,0]])
            self. S_z = np.array([[1,0,0],
                            [0,0,0],
                            [0,0,-1]])

            self.I = np.array([[1,0,0],
                [0,1,0],
                [0,0,1]])


        elif S == 1/2:
            #matrices for S = 1/2
            self.S_plus = np.array([[0,1],
                                    [0,0]])

            self.S_minus = np.array([[0,0],
                                    [1,0]])

            self.S_z = 1/2 * np.array([[1,0],
                                    [0,-1]])
            self.I = np.array([[1,0],
                                [0,1]])

        # Sparse copies for use in Hamiltonian construction
        self.S_plus_sp = scipy.sparse.csr_matrix(self.S_plus)
        self.S_minus_sp = scipy.sparse.csr_matrix(self.S_minus)
        self.S_z_sp = scipy.sparse.csr_matrix(self.S_z)
        self.I_sp = scipy.sparse.csr_matrix(self.I)

    def S_site(self, index, S_op):
        #Using tensor product to calculate S_i matrix (sparse version)
        N = self.size_of_system
        d = S_op.shape[0]
        left = scipy.sparse.eye(d**index, format='csr')
        right = scipy.sparse.eye(d**(N - 1 - index), format='csr')
        return scipy.sparse.kron(scipy.sparse.kron(left, S_op, format='csr'), right, format='csr')

    def S_z_operator(self):
        #calculating S_z operator as sum S_z_1 + S_z_2 + ... + S_z_N
        S_z_operator = 0
        #print(self.S_z)
        for i in range(self.size_of_system):
            S_z_operator  += self.S_site(i, self.S_z_sp)

        #print(S_z_operator)
        return S_z_operator

    def calc_Sz(self, eigenvector):
        # Calculate the conjugate transpose of the eigenvector
        eigen_dagger = np.conj(eigenvector.T)
        # Calculate the expectation value of S_z
        S_z_op = self.S_z_operator()
        if scipy.sparse.issparse(S_z_op):
            S_z_op = S_z_op.toarray()
        Sz_total = np.dot(eigen_dagger, np.dot(S_z_op, eigenvector))
        return Sz_total

    def calculate_basis(self):
        N = self.size_of_system
        #for bais s=1/2 -> (up - True, down - False)
        #for bais s=1 -> (-1,0,1)

        if self.S == 1/2:
            self.list_spins = [1/2,-1/2]
        elif self.S == 1:
            self.list_spins = [-1,0,1]

        for i in range(N):
            self.possible_basis.append(self.list_spins)

        #whole basis
        #basis_s_z = []
        self.basis = list(product(*self.possible_basis))
        self.basis_s_z = self.basis[:]

        #self.basis = list(map(lambda x: list(x), self.basis))
        #print(self.basis)

        for i in range(len(self.basis_s_z)):
            self.basis_s_z[i] = sum(self.basis_s_z[i])

        #print(self.basis)
        #all possible spin combinations
        self.list_of_spins = sorted(list(set(self.basis_s_z)),reverse=True)
        #print(self.list_of_spins)

        return self.basis, self.basis_s_z, self.list_of_spins


    def subsystems_fixed_s_z(self, spin_basis,size_of_sub_A,size_of_sub_B):
        #function for calculating bases of subsystems A and B

        assert size_of_sub_A + size_of_sub_B == self.size_of_system, \
            f"size_of_sub_A ({size_of_sub_A}) + size_of_sub_B ({size_of_sub_B}) != N ({self.size_of_system})"

        #DIVISION OF A AND B HERE IS VERY CRUCIAL FOR THE RHO CALCULATION

        subsystem_A = list(set(map(lambda x: x[:size_of_sub_A], spin_basis)))
        subsystem_B = list(set(map(lambda x: x[size_of_sub_A:], spin_basis)))

        subsystem_A_beta = list(map(lambda x: x[:size_of_sub_A], spin_basis))
        subsystem_B_beta = list(map(lambda x: x[size_of_sub_A:], spin_basis))


        #subsystem_A = list(set(map(lambda x: x[:int(len(x)/size_of_sub_A)], spin_basis)))
        #subsystem_B = list(set(map(lambda x: x[int(len(x)/size_of_sub_B):], spin_basis)))

        #subsystem_A_beta = list(map(lambda x: x[:int(len(x)/size_of_sub_A)], spin_basis))
        #subsystem_B_beta = list(map(lambda x: x[int(len(x)/size_of_sub_B):], spin_basis))

        #print(subsystem_A_beta)
        #print(subsystem_B_beta)
        #print(f"Basis for subsystem A: {subsystem_A}")
        #print(f"Basis for subsystem B: {subsystem_B}")

        new_basis = []

        for k in spin_basis:
            #k_A = k[:int(len(k)/size_of_sub_A)]
            #print(f"This k_A {k_A}")
            #k_B = k[int(len(k)/size_of_sub_B):]

            k_A = k[:size_of_sub_A]
            #print(f"This k_A {k_A}")
            k_B = k[size_of_sub_A:]
            #print(f"This k_B {k_B}")

            i = subsystem_A.index(k_A)
            j = subsystem_B.index(k_B)
            #print(f"This is i {i}")
            #print(f"This is j {j}")
            if (i,j) not in new_basis:
                new_basis.append((i,j))

        #print("This is new basis: ", new_basis)

        return subsystem_A, subsystem_B, new_basis


    def create_Hamiltonian(self, adjMatrix):
        #definition of S matrices and diagonalization

        adjMatrix = np.asarray(adjMatrix)
        assert adjMatrix.ndim == 2, f"adjMatrix must be 2D, got {adjMatrix.ndim}D"
        assert adjMatrix.shape[0] == adjMatrix.shape[1], \
            f"adjMatrix must be square, got shape {adjMatrix.shape}"
        assert adjMatrix.shape[0] == self.size_of_system, \
            f"adjMatrix size ({adjMatrix.shape[0]}) != N ({self.size_of_system})"

        N = self.size_of_system
        d = len(self.I)
        dim = d**N

        # Precompute sparse site operators for all sites
        Sp = [self.S_site(k, self.S_plus_sp) for k in range(N)]
        Sm = [self.S_site(k, self.S_minus_sp) for k in range(N)]
        Sz = [self.S_site(k, self.S_z_sp) for k in range(N)]

        self.H = scipy.sparse.csr_matrix((dim, dim), dtype=float)

        #using adjacency matrix to define neighbouring sites
        for i in range(N):
            for j in range(N):
                if adjMatrix[j][i] == 1:
                    self.H += 0.5 * (Sp[j].dot(Sm[i]) + Sm[j].dot(Sp[i])) \
                            + Sz[j].dot(Sz[i])

        #for i in range(len(self.H)):
                #self.H[i][i] += np.random.random()*10e-8
        #print(self.H)
        print("Len of Hamiltonian: ", dim)


    def create_Hamiltonian_AKLT(self, adjMatrix):
        # AKLT Hamiltonian: H = sum_<i,j> [ S_i·S_j + 1/3 (S_i·S_j)^2 ]
        # For S=1 this is a projector onto the S=2 subspace of each bond.
        # For S=1/2 the biquadratic term reduces to const + linear S·S (equivalent to shifted Heisenberg).

        adjMatrix = np.asarray(adjMatrix)
        assert adjMatrix.ndim == 2, f"adjMatrix must be 2D, got {adjMatrix.ndim}D"
        assert adjMatrix.shape[0] == adjMatrix.shape[1], \
            f"adjMatrix must be square, got shape {adjMatrix.shape}"
        assert adjMatrix.shape[0] == self.size_of_system, \
            f"adjMatrix size ({adjMatrix.shape[0]}) != N ({self.size_of_system})"

        N = self.size_of_system
        d = len(self.I)
        dim = d**N

        # Precompute sparse site operators for all sites
        Sp = [self.S_site(k, self.S_plus_sp) for k in range(N)]
        Sm = [self.S_site(k, self.S_minus_sp) for k in range(N)]
        Sz = [self.S_site(k, self.S_z_sp) for k in range(N)]

        self.H = scipy.sparse.csr_matrix((dim, dim), dtype=float)

        # Uses same adjacency matrix convention as create_Hamiltonian:
        # each bond appears once in adjMatrix (directed), no double-counting
        for i in range(N):
            for j in range(N):
                if adjMatrix[j][i] == 1:
                    # h_ij = S_i · S_j = 1/2 (S+_j S-_i + S-_j S+_i) + Sz_j Sz_i
                    h_ij = 0.5 * (Sp[j].dot(Sm[i]) + Sm[j].dot(Sp[i])) \
                         + Sz[j].dot(Sz[i])
                    self.H += h_ij + (1.0/3.0) * h_ij.dot(h_ij)

        print("Len of Hamiltonian: ", dim)

    def eig_diagonalize(self,A):
        #fucntion for diagonalization with sorting eigenvalues and rewriting eigenvectors as a list
        if scipy.sparse.issparse(A):
            A = A.toarray()
        eigenValues, eigenVectors = np.linalg.eig(A)
        idx = eigenValues.argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:,idx]
        return eigenValues, eigenVectors

    def block_Hamiltonian(self,iterator):

        target_spin = self.list_of_spins[iterator]
        row_indices = [i for i, sz in enumerate(self.basis_s_z) if sz == target_spin]
        spin_basis = [self.basis[i] for i in row_indices]

        H_dense = self.H.toarray() if scipy.sparse.issparse(self.H) else self.H
        block_H_spin = H_dense[np.ix_(row_indices, row_indices)]

        print(f"Block S_z={target_spin}: size {block_H_spin.shape[0]}x{block_H_spin.shape[1]}")

        if self.size_of_system <= 6:
            print(block_H_spin)

        energies, vectors = self.eig_diagonalize(block_H_spin)

        return energies, vectors, spin_basis

    def calculate_rho_system(self,psi0):
        #old functions for calulating rho for H not divdied into fixed S_z blocks

        #(2S+1)**N_sys
        size_of_subsystem = len(self.I)**(int(self.size_of_system/2))
        psi0 = psi0.reshape([size_of_subsystem, -1], order="C")
        #print("2 This is psi0: \n", psi0)
        rho = np.dot(psi0, psi0.conj().transpose())

        return rho

    def calculate_rho_env(self,psi0):
        #old functions for calulating rho for H not divdied into fixed S_z blocks

        size_of_subsystem = len(self.I)**(int(self.size_of_system/2))
        psi0 = psi0.reshape([size_of_subsystem, -1], order="C")
        #print("2 This is psi0: \n", psi0)
        rho = np.dot(psi0.conj().transpose(), psi0)

        return rho

    def calculate_entropy(self,rho_reduced,n):

        #Here depending if s = 1/2 or s = 1 you need to change the base of log

        #n - number of spins in the subsystem
        d = int(2*self.S + 1)
        eigen_rho, vectors = self.eig_diagonalize(rho_reduced)

        #entropy = -sum(eigen_rho*np.log(eigen_rho, where=0<eigen_rho, out=0.0*eigen_rho))
        #eigen_rho_nonzero = eigen_rho[(eigen_rho > 10e-8) & (eigen_rho < 1.0)]
        #entropy = -np.sum(eigen_rho_nonzero * np.log2(eigen_rho_nonzero))

        entropy = 0
        for i in range(len(eigen_rho)):
            #print(eigen_rho[i])
            if eigen_rho[i] <= ZERO_THRESHOLD:
                entropy += 0.0

            elif abs(eigen_rho[i] - 1.0) < ZERO_THRESHOLD:
                entropy += 0.0

            else:
                entropy += -(eigen_rho[i]*np.log2(eigen_rho[i]))
                #entropy += -(eigen_rho[i]*math.log(eigen_rho[i],3))

        #return entropy, eigen_rho
        max_entropy = n * np.log2(d)
        if max_entropy == 0:
            entropy_normalized = entropy
        else:
            entropy_normalized = entropy / max_entropy

        return entropy_normalized, entropy, eigen_rho

    def calculate_S_z(self,vectors):
        #S_z value calculated as inner product of S_z operator and eigenvectors of H
        S_z_total = []
        for i in range(len(vectors)):
            S_z_total.append(self.calc_Sz(vectors[:,i]))

        return S_z_total
