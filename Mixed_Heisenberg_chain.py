# Generalized mixed-spin Heisenberg model
# Each site can have a different spin value (e.g. [1, 1/2, 1/2, 1])
# Couplings defined by three N x N matrices: J_x, J_y, J_z
#
# H = (1/2) sum_{i!=j} [ (Jx_ij - Jy_ij)/4 S+_i S+_j
#                       + (Jx_ij + Jy_ij)/4 S-_i S+_j
#                       + (Jx_ij + Jy_ij)/4 S+_i S-_j
#                       + (Jx_ij - Jy_ij)/4 S-_i S-_j
#                       + Jz_ij Sz_i Sz_j ]
#
# For isotropic Heisenberg: J_x = J_y = J_z = symmetric adjacency matrix
# Note: Sz block diagonalization only valid when J_x == J_y (otherwise S+S+ and S-S- terms mix Sz sectors)

import numpy as np
from itertools import product
import scipy.sparse

ZERO_THRESHOLD  = 1e-8
TRACE_TOLERANCE = 1e-12


def build_spin_operators(s):
    """Build S+, S-, Sz, I matrices for arbitrary spin s.
    Basis ordered: m = s, s-1, ..., -s (descending).
    Uses: S+|s,m> = sqrt(s(s+1) - m(m+1)) |s,m+1>
          S-|s,m> = sqrt(s(s+1) - m(m-1)) |s,m-1>
    """
    dim = int(2*s + 1)
    S_plus = np.zeros((dim, dim))
    S_minus = np.zeros((dim, dim))
    S_z = np.zeros((dim, dim))

    for idx in range(dim):
        m = s - idx
        S_z[idx, idx] = m
        # S+ raises m -> m+1 (idx -> idx-1)
        if idx > 0:
            S_plus[idx-1, idx] = np.sqrt(s*(s+1) - m*(m+1))
        # S- lowers m -> m-1 (idx -> idx+1)
        if idx < dim - 1:
            S_minus[idx+1, idx] = np.sqrt(s*(s+1) - m*(m-1))

    return S_plus, S_minus, S_z, np.eye(dim)


class MixedHeisenberg:

    def __init__(self, spin_sizes):
        """
        spin_sizes: list of spin values per site, e.g. [1, 0.5, 0.5, 1]
        """
        self.spin_sizes = list(spin_sizes)
        self.size_of_system = len(spin_sizes)
        self.dims = [int(2*s + 1) for s in spin_sizes]
        self.total_dim = int(np.prod(self.dims))

        self.H = None
        self.basis = []
        self.basis_s_z = []
        self.list_of_spins = []

        # Per-site sparse operators
        self.site_ops = []
        for s in spin_sizes:
            Sp, Sm, Sz, I = build_spin_operators(s)
            self.site_ops.append({
                'S_plus': scipy.sparse.csr_matrix(Sp),
                'S_minus': scipy.sparse.csr_matrix(Sm),
                'S_z': scipy.sparse.csr_matrix(Sz),
                'I': scipy.sparse.csr_matrix(I),
            })

    def S_site(self, index, S_op):
        """Build full-space operator: I_0 x ... x S_op x ... x I_{N-1}
        with correct (variable) dimensions per site."""
        N = self.size_of_system
        left_dim = int(np.prod(self.dims[:index])) if index > 0 else 1
        right_dim = int(np.prod(self.dims[index+1:])) if index < N-1 else 1

        left = scipy.sparse.eye(left_dim, format='csr')
        right = scipy.sparse.eye(right_dim, format='csr')
        return scipy.sparse.kron(scipy.sparse.kron(left, S_op, format='csr'), right, format='csr')

    def S_z_operator(self):
        """Total S_z = sum_i S_z^(i)"""
        Sz_op = scipy.sparse.csr_matrix((self.total_dim, self.total_dim))
        for i in range(self.size_of_system):
            Sz_op += self.S_site(i, self.site_ops[i]['S_z'])
        return Sz_op

    def calc_Sz(self, eigenvector):
        eigen_dagger = np.conj(eigenvector.T)
        S_z_op = self.S_z_operator()
        if scipy.sparse.issparse(S_z_op):
            S_z_op = S_z_op.toarray()
        return np.dot(eigen_dagger, np.dot(S_z_op, eigenvector))

    def calculate_basis(self):
        """Generate all product basis states and their total S_z values.
        Each site k has m_k in {s_k, s_k-1, ..., -s_k}."""
        N = self.size_of_system
        possible_m = []
        for k in range(N):
            s = self.spin_sizes[k]
            dim = self.dims[k]
            possible_m.append([s - idx for idx in range(dim)])

        self.basis = list(product(*possible_m))
        self.basis_s_z = [sum(config) for config in self.basis]
        self.list_of_spins = sorted(set(self.basis_s_z), reverse=True)

        return self.basis, self.basis_s_z, self.list_of_spins

    def create_Hamiltonian(self, J_x, J_y, J_z):
        """
        Build H using the general anisotropic formula with coupling matrices.

        J_x, J_y, J_z: N x N symmetric coupling matrices.
        For isotropic nearest-neighbor Heisenberg: J_x = J_y = J_z = symmetric adjacency matrix.
        """
        N = self.size_of_system
        J_x = np.asarray(J_x, dtype=float)
        J_y = np.asarray(J_y, dtype=float)
        J_z = np.asarray(J_z, dtype=float)

        for name, J in [("J_x", J_x), ("J_y", J_y), ("J_z", J_z)]:
            assert J.shape == (N, N), \
                f"{name} shape {J.shape} does not match chain size N={N}"
            assert np.allclose(J, J.T), \
                f"{name} must be symmetric (Hermitian for real couplings)"

        dim = self.total_dim

        # Precompute full-space operators for each site
        Sp = [self.S_site(k, self.site_ops[k]['S_plus']) for k in range(N)]
        Sm = [self.S_site(k, self.site_ops[k]['S_minus']) for k in range(N)]
        Sz = [self.S_site(k, self.site_ops[k]['S_z']) for k in range(N)]

        self.H = scipy.sparse.csr_matrix((dim, dim), dtype=float)

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                J_pp = 0.25 * (J_x[i, j] - J_y[i, j])  # coeff of S+_i S+_j
                J_mp = 0.25 * (J_x[i, j] + J_y[i, j])  # coeff of S-_i S+_j and S+_i S-_j
                J_zz = J_z[i, j]                       # coeff of Sz_i Sz_j

                if abs(J_pp) > 0:
                    self.H += J_pp * Sp[i].dot(Sp[j])
                    self.H += J_pp * Sm[i].dot(Sm[j])
                if abs(J_mp) > 0:
                    self.H += J_mp * Sm[i].dot(Sp[j])
                    self.H += J_mp * Sp[i].dot(Sm[j])
                if abs(J_zz) > 0:
                    self.H += J_zz * Sz[i].dot(Sz[j])
        
        #+1/2 before the hamiltonian!
        self.H *= 0.5

        print(f"Hamiltonian dimension: {dim}")

    def create_Hamiltonian_AKLT(self, J_x, J_y, J_z):
        """AKLT: H = sum_{<i,j>} [h_ij + 1/3 h_ij^2] where h_ij = S_i . S_j.
        Bonds determined by upper triangle of J matrices (i < j with nonzero entry)."""
        N = self.size_of_system
        J_x = np.asarray(J_x, dtype=float)
        J_y = np.asarray(J_y, dtype=float)
        J_z = np.asarray(J_z, dtype=float)

        for name, J in [("J_x", J_x), ("J_y", J_y), ("J_z", J_z)]:
            assert J.shape == (N, N), \
                f"{name} shape {J.shape} does not match chain size N={N}"

        dim = self.total_dim

        Sp = [self.S_site(k, self.site_ops[k]['S_plus']) for k in range(N)]
        Sm = [self.S_site(k, self.site_ops[k]['S_minus']) for k in range(N)]
        Sz = [self.S_site(k, self.site_ops[k]['S_z']) for k in range(N)]

        self.H = scipy.sparse.csr_matrix((dim, dim), dtype=float)

        for i in range(N):
            for j in range(i+1, N):
                if abs(J_x[i, j]) > 0 or abs(J_y[i, j]) > 0 or abs(J_z[i, j]) > 0:
                    #+1/2 before the hamiltonian!
                    h_ij = 0.5 * (Sp[i].dot(Sm[j]) + Sm[i].dot(Sp[j])) \
                            + Sz[i].dot(Sz[j])
                    self.H += h_ij + (1.0/3.0) * h_ij.dot(h_ij)

        print(f"Hamiltonian dimension: {dim}")

    def eig_diagonalize(self, A):
        if scipy.sparse.issparse(A):
            A = A.toarray()
        eigenValues, eigenVectors = np.linalg.eig(A)
        idx = eigenValues.argsort()
        eigenValues = eigenValues[idx]
        eigenVectors = eigenVectors[:, idx]
        return eigenValues, eigenVectors

    def block_Hamiltonian(self, iterator):
        target_spin = self.list_of_spins[iterator]
        row_indices = [i for i, sz in enumerate(self.basis_s_z)
                        if abs(sz - target_spin) < 1e-12]
        spin_basis = [self.basis[i] for i in row_indices]

        H_dense = self.H.toarray() if scipy.sparse.issparse(self.H) else self.H
        block_H_spin = H_dense[np.ix_(row_indices, row_indices)]

        print(f"Block S_z={target_spin}: size {block_H_spin.shape[0]}x{block_H_spin.shape[1]}")

        if self.size_of_system <= 6:
            print(block_H_spin)

        if np.iscomplexobj(block_H_spin) and np.allclose(block_H_spin.imag, 0):
            block_H_spin = block_H_spin.real

        energies, vectors = self.eig_diagonalize(block_H_spin)

        return energies, vectors, spin_basis

    # --- Translation operator and momentum subspaces ---
    # Requires: uniform spin chain (all spins equal) + PBC Hamiltonian
    # [T, H] = 0 only with PBC; [T, Sz_total] = 0 always for uniform chains

    def build_translation_operator(self):
        """Build cyclic translation operator T as a permutation matrix.
        T|s_1, s_2, ..., s_N> = |s_N, s_1, ..., s_{N-1}>
        """
        assert len(set(self.spin_sizes)) == 1, \
            "Translation operator requires uniform spin chain (all spins equal)"
        assert len(self.basis) > 0, "Call calculate_basis() first"

        config_to_idx = {config: idx for idx, config in enumerate(self.basis)}

        dim = self.total_dim
        rows, cols = [], []
        for idx, config in enumerate(self.basis):
            shifted = (config[-1],) + config[:-1]
            new_idx = config_to_idx[shifted]
            rows.append(new_idx)
            cols.append(idx)

        data = np.ones(dim, dtype=float)
        self.T = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(dim, dim))
        print(f"Translation operator built: {dim}x{dim}")
        return self.T

    def _get_sz_block_indices(self, iterator):
        """Return row indices and spin_basis for Sz sector `iterator`."""
        target_spin = self.list_of_spins[iterator]
        row_indices = [i for i, sz in enumerate(self.basis_s_z)
                       if abs(sz - target_spin) < 1e-12]
        spin_basis = [self.basis[i] for i in row_indices]
        return row_indices, spin_basis

    def block_translation_operator(self, iterator):
        """Extract T block for Sz sector `iterator`.
        Returns: T_Sz (dense matrix), row_indices"""
        assert hasattr(self, 'T') and self.T is not None, \
            "Call build_translation_operator() first"

        row_indices, _ = self._get_sz_block_indices(iterator)
        T_dense = self.T.toarray() if scipy.sparse.issparse(self.T) else self.T
        block_T = T_dense[np.ix_(row_indices, row_indices)]

        return block_T, row_indices

    def momentum_diagonalize(self, iterator):
        """For Sz sector `iterator`:
        1. Extract H_Sz and T_Sz blocks
        2. Diagonalize T_Sz -> momentum eigenstates forming U_Sz
        3. Group eigenvectors by k
        4. Rotate: H_k = U_Sz^dag H_Sz U_Sz (block-diagonal by k)
        5. Diagonalize each k-block

        Returns: results, H_rotated, U_Sz, k_sorted
          results: list of (k_value, energies_array) per momentum sector
          H_rotated: full rotated H in (Sz,k) basis
          U_Sz: unitary matrix of T eigenvectors
          k_sorted: k label for each column of U_Sz
        """
        N = self.size_of_system
        target_spin = self.list_of_spins[iterator]

        # Get H_Sz block
        row_indices, spin_basis = self._get_sz_block_indices(iterator)
        H_dense = self.H.toarray() if scipy.sparse.issparse(self.H) else self.H
        H_Sz = H_dense[np.ix_(row_indices, row_indices)]
        if np.iscomplexobj(H_Sz) and np.allclose(H_Sz.imag, 0):
            H_Sz = H_Sz.real

        # Get T_Sz block
        block_T, _ = self.block_translation_operator(iterator)

        # Diagonalize T_Sz
        T_evals, T_evecs = np.linalg.eig(block_T)

        # Extract k from eigenvalues e^{ik}
        k_values = np.angle(T_evals)

        # Round to allowed k = 2*pi*m / N, m = 0, 1, ..., N-1
        # shifted to [-pi, pi)
        allowed_k = np.array([2 * np.pi * m / N for m in range(N)])
        allowed_k = (allowed_k + np.pi) % (2 * np.pi) - np.pi

        k_assigned = np.zeros_like(k_values)
        for idx, kv in enumerate(k_values):
            diffs = np.abs((allowed_k - kv + np.pi) % (2 * np.pi) - np.pi)
            k_assigned[idx] = allowed_k[np.argmin(diffs)]

        # Sort eigenvectors by k for clean block structure
        sort_order = np.argsort(k_assigned)
        U_Sz = T_evecs[:, sort_order]
        k_sorted = k_assigned[sort_order]

        # Rotate H: U^dag H U
        H_rotated = U_Sz.conj().T @ H_Sz @ U_Sz

        # Extract and diagonalize each k-block
        unique_k = np.unique(np.round(k_sorted, 10))
        results = []
        for k in unique_k:
            k_mask = np.abs(k_sorted - k) < 1e-10
            k_indices = np.where(k_mask)[0]

            H_k = H_rotated[np.ix_(k_indices, k_indices)]
            # Enforce Hermiticity for numerical stability
            H_k = 0.5 * (H_k + H_k.conj().T)

            if H_k.shape[0] == 1:
                e_k = np.array([np.real(H_k[0, 0])])
            else:
                e_k = np.sort(np.real(np.linalg.eigvalsh(H_k)))

            results.append((k, e_k))

        print(f"  S_z={target_spin}: {len(row_indices)} states -> "
              f"{len(unique_k)} k-sectors: "
              + ", ".join(f"k={k:.4f}({len(e)})" for k, e in results))

        return results, H_rotated, U_Sz, k_sorted

    def momentum_diagonalize_all(self):
        """Run momentum diagonalization for all Sz sectors.
        Returns: all_results = list of (Sz, k, energies) tuples
        """
        all_results = []
        for i in range(len(self.list_of_spins)):
            sz = self.list_of_spins[i]
            results, _, _, _ = self.momentum_diagonalize(i)
            for k, energies in results:
                all_results.append((sz, k, energies))
        return all_results

    # --- Entropy methods (unchanged from original Heisenberg class) ---

    def subsystems_fixed_s_z(self, spin_basis, size_of_sub_A, size_of_sub_B):
        assert size_of_sub_A + size_of_sub_B == self.size_of_system, \
            f"size_of_sub_A ({size_of_sub_A}) + size_of_sub_B ({size_of_sub_B}) != N ({self.size_of_system})"

        subsystem_A = list(set(map(lambda x: x[:size_of_sub_A], spin_basis)))
        subsystem_B = list(set(map(lambda x: x[size_of_sub_A:], spin_basis)))

        new_basis = []
        for k in spin_basis:
            k_A = k[:size_of_sub_A]
            k_B = k[size_of_sub_A:]
            i = subsystem_A.index(k_A)
            j = subsystem_B.index(k_B)
            if (i, j) not in new_basis:
                new_basis.append((i, j))

        return subsystem_A, subsystem_B, new_basis

    def calculate_entropy(self, rho_reduced, n):
        #n - number of spins in the subsystem
        d = max(self.dims)
        eigen_rho, vectors = self.eig_diagonalize(rho_reduced)

        entropy = 0
        for i in range(len(eigen_rho)):
            if eigen_rho[i] <= ZERO_THRESHOLD:
                entropy += 0.0
            elif abs(eigen_rho[i] - 1.0) < ZERO_THRESHOLD:
                entropy += 0.0
            else:
                entropy += -(eigen_rho[i]*np.log2(eigen_rho[i]))

        max_entropy = n * np.log2(d)
        if max_entropy == 0:
            entropy_normalized = entropy
        else:
            entropy_normalized = entropy / max_entropy

        return entropy_normalized, entropy, eigen_rho
