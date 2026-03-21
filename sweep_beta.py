"""
Sweep beta from 0 (Heisenberg) to 1/3 (AKLT) for S=1 chain.
H(beta) = sum_{<i,j>} [h_ij + beta * h_ij^2]

For each beta: Sz block diagonalization, entropy calculation, save results.
"""

import numpy as np
import Mixed_Heisenberg_chain
import math
import os
import time
import argparse

TRACE_TOLERANCE = 1e-12


def run_single_beta(H, beta, spins, size_of_sub_A, size_of_sub_B):
    """Diagonalize H(beta) in Sz <= 0 blocks, mirror for Sz > 0.
    Exploits Sz <-> -Sz symmetry: same energies, same entropies.
    Returns: all_energies, entropy_norm, entropy_raw, s_z_number, sum_lambdas"""
    H.set_hamiltonian_from_beta(beta)

    all_energies = []
    entropy_norm_all = []
    entropy_raw_all = []
    s_z_all = []
    sum_lambdas_all = []

    # spins is sorted descending: [S_max, ..., 0, ..., -S_max]
    # Only compute Sz <= 0 sectors
    for i, spin in enumerate(spins):
        if spin > 0:
            continue

        energies, vectors, spin_basis = H.block_Hamiltonian(i, verbose=False)

        subsystem_A, subsystem_B, new_basis = H.subsystems_fixed_s_z(
            spin_basis, size_of_sub_A, size_of_sub_B)

        sector_energies = []
        sector_ent_norm = []
        sector_ent_raw = []
        sector_sum_lam = []

        for j in range(len(energies)):
            psi = np.zeros((len(subsystem_A), len(subsystem_B)), dtype=complex)
            for k, v in enumerate(vectors[:, j]):
                psi[new_basis[k][0]][new_basis[k][1]] = v

            rho = np.dot(psi, psi.conj().T)

            if abs(np.trace(rho) - 1.0) > TRACE_TOLERANCE:
                print(f"  Warning: Tr(rho)={np.trace(rho):.6f} for "
                      f"Sz={spin}, E={energies[j]:.6f}, beta={beta:.4f}")

            entropy_norm, entropy_raw, eigen_rho = H.calculate_entropy(rho, size_of_sub_A)

            sector_energies.append(np.real(energies[j]))
            sector_ent_norm.append(np.real(entropy_norm))
            sector_ent_raw.append(np.real(entropy_raw))
            sector_sum_lam.append(np.real(sum(eigen_rho)))

        # Add this Sz <= 0 sector
        all_energies.extend(sector_energies)
        entropy_norm_all.extend(sector_ent_norm)
        entropy_raw_all.extend(sector_ent_raw)
        s_z_all.extend([spin] * len(sector_energies))
        sum_lambdas_all.extend(sector_sum_lam)

        # Mirror to Sz > 0 (skip Sz=0 to avoid double counting)
        if spin < 0:
            all_energies.extend(sector_energies)
            entropy_norm_all.extend(sector_ent_norm)
            entropy_raw_all.extend(sector_ent_raw)
            s_z_all.extend([-spin] * len(sector_energies))
            sum_lambdas_all.extend(sector_sum_lam)

    return all_energies, entropy_norm_all, entropy_raw_all, s_z_all, sum_lambdas_all


def main():
    parser = argparse.ArgumentParser(description="Sweep beta: Heisenberg (beta=0) to AKLT (beta=1/3)")
    parser.add_argument("-N", "--sites", type=int, default=10)
    parser.add_argument("-S", "--spin", type=float, default=1.0)
    parser.add_argument("-b", "--boundary", type=str, default="OBC", choices=["PBC", "OBC"])
    parser.add_argument("--beta-min", type=float, default=0.0)
    parser.add_argument("--beta-max", type=float, default=1.0/3.0)
    parser.add_argument("--beta-step", type=float, default=0.01)
    args = parser.parse_args()

    N = args.sites
    S = args.spin
    boundary = args.boundary

    if S == 0.5:
        S_save = "1_2"
    else:
        S_save = str(int(S))

    # Build chain
    spin_sizes = [S] * N
    H = Mixed_Heisenberg_chain.MixedHeisenberg(spin_sizes)
    basis_H, basis_H_s_z, spins = H.calculate_basis()

    # Coupling matrix
    adjMatrix = np.eye(N, k=1, dtype=int)
    if boundary == "PBC":
        adjMatrix[-1][0] = 1
    J = np.clip((adjMatrix + adjMatrix.T).astype(float), 0, 1)

    # Precompute bilinear + biquadratic
    print(f"Building H_bilinear and H_biquadratic for N={N}, S={S}, {boundary}...")
    t0 = time.time()
    H.build_bilinear_biquadratic(J, J, J)
    print(f"  Done in {time.time() - t0:.1f}s")

    size_of_sub_A = math.ceil(N / 2)
    size_of_sub_B = N - size_of_sub_A

    # Beta values
    betas = np.arange(args.beta_min, args.beta_max + args.beta_step / 2, args.beta_step)
    betas = np.round(betas, 6)

    # Output directory
    dir_name = f"./results/results_beta_sweep_{N}_sites_{S_save}_{boundary}"
    os.makedirs(dir_name, exist_ok=True)

    print(f"\nSweeping beta from {betas[0]:.4f} to {betas[-1]:.4f} "
          f"({len(betas)} values, step={args.beta_step})")
    sz0_count = len([s for s in basis_H_s_z if abs(s - 0) < 1e-12])
    print(f"Sz blocks: {len(spins)} sectors, largest (Sz=0): {sz0_count}")
    print(f"Output: {dir_name}/")

    total_t0 = time.time()

    for idx, beta in enumerate(betas):
        t1 = time.time()
        print(f"\n[{idx+1}/{len(betas)}] beta={beta:.4f} ...", end="", flush=True)

        energies, ent_norm, ent_raw, sz, sum_lam = run_single_beta(
            H, beta, spins, size_of_sub_A, size_of_sub_B)

        # Save results
        results = np.column_stack([energies, ent_norm, ent_raw, sz, sum_lam])
        beta_str = f"{beta:.4f}"
        fname = f"{dir_name}/results_beta_{beta_str}_{S_save}_{N}.csv"
        np.savetxt(fname, results, delimiter=',',
                   header="Energy, Entropy_normalized, Entropy_raw, S_z, Sum_lambdas")

        elapsed = time.time() - t1
        print(f" {len(energies)} states, {elapsed:.1f}s")

    total_elapsed = time.time() - total_t0
    print(f"\nDone! Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"Results saved in: {dir_name}/")


if __name__ == "__main__":
    main()
