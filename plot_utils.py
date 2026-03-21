import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.cm as cm


# function to plot Entropy against eigenvalue with cmap of Sz
# mode='leq' filters S_z <= sz_filter, mode='eq' filters S_z == sz_filter
def plot_entropy_against_eigenvalue(filepath, figsize=(8, 6), sz_filter=0, mode='leq', figpath=None, entropy='normalized', yticks=None, dpi=100, n_lowest=None):
    # entropy: 'normalized' (col 1) or 'raw' (col 2)
    try:
        data = np.loadtxt(filepath, delimiter=',')
    except ValueError:
        data = np.loadtxt(filepath)
    energies = data[:, 0]
    entropy_col = 1 if entropy == 'normalized' else 2
    entropies = data[:, entropy_col]
    sz = data[:, 3]

    if mode == 'eq':
        mask = np.isclose(sz, sz_filter)
    else:
        mask = sz <= sz_filter
    energies = energies[mask]
    entropies = entropies[mask]
    sz = sz[mask]

    if n_lowest is not None:
        keep = np.zeros(len(sz), dtype=bool)
        for sz_val in np.unique(sz):
            sz_mask = np.isclose(sz, sz_val)
            indices = np.where(sz_mask)[0]
            sorted_indices = indices[np.argsort(energies[indices])][:n_lowest]
            keep[sorted_indices] = True
        energies = energies[keep]
        entropies = entropies[keep]
        sz = sz[keep]

    # use integer S_z if all values are whole numbers
    all_integer = np.all(np.isclose(sz, np.round(sz)))
    if all_integer:
        sz = np.round(sz).astype(int)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sc = ax.scatter(energies, entropies, c=sz, cmap='viridis', edgecolors='k', linewidths=0.01,
                    vmin=sz.min(), vmax=sz.max())
    if mode != 'eq':
        cbar = plt.colorbar(sc, ax=ax)
        if all_integer:
            cbar.set_ticks(np.arange(sz.min(), sz.max() + 1))
        cbar.set_label(r'$S_z$')
    ax.set_xlabel(r'Eigenvalue $E$')
    ax.set_ylabel(r'Entropy $S$')
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(direction='in', which='both')
    plt.tight_layout()
    if figpath:
        plt.savefig(figpath, dpi=dpi, bbox_inches='tight')
    plt.show()


def plot_momentum_entropy(filepath, figsize=(8, 6), sz_filter=None, figpath=None, entropy='normalized', yticks=None, dpi=100, n_lowest=None):
    """Plot entropy vs energy from momentum CSV, colored by k/pi.

    Momentum CSV columns: Energy(0), S_z(1), k(2), k_over_pi(3), Entropy_norm(4), Entropy_raw(5)
    sz_filter: if set, show only states with that S_z value.
    """
    try:
        data = np.loadtxt(filepath, delimiter=',')
    except ValueError:
        data = np.loadtxt(filepath)
    energies = data[:, 0]
    sz = data[:, 1]
    k_over_pi = data[:, 3]
    entropy_col = 4 if entropy == 'normalized' else 5
    entropies = data[:, entropy_col]

    if sz_filter is not None:
        mask = np.isclose(sz, sz_filter)
        energies = energies[mask]
        entropies = entropies[mask]
        k_over_pi = k_over_pi[mask]
        sz = sz[mask]

    if n_lowest is not None:
        sorted_indices = np.argsort(energies)[:n_lowest]
        energies = energies[sorted_indices]
        entropies = entropies[sorted_indices]
        k_over_pi = k_over_pi[sorted_indices]
        sz = sz[sorted_indices]

    # Spread degenerate points horizontally so each dot is visible
    e_plot = energies.copy()
    e_round = np.round(energies, 6)
    s_round = np.round(entropies, 6)
    spread = 0.06 * (energies.max() - energies.min()) if energies.max() != energies.min() else 0.06
    visited = {}
    for i in range(len(e_plot)):
        key = (e_round[i], s_round[i])
        if key not in visited:
            visited[key] = []
        visited[key].append(i)
    for key, indices in visited.items():
        n = len(indices)
        if n > 1:
            offsets = np.linspace(-spread * (n - 1) / 2, spread * (n - 1) / 2, n)
            for j, idx in enumerate(indices):
                e_plot[idx] += offsets[j]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sc = ax.scatter(e_plot, entropies, c=k_over_pi, cmap='coolwarm', edgecolors='k', linewidths=0.3,
                    vmin=k_over_pi.min(), vmax=k_over_pi.max())
    cbar = plt.colorbar(sc, ax=ax)
    unique_k = np.unique(np.round(k_over_pi, 6))
    cbar.set_ticks(unique_k)
    cbar.set_label(r'$k/\pi$')
    ax.set_xlabel(r'Eigenvalue $E$')
    ax.set_ylabel(r'Entropy $S$')
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.tick_params(direction='in', which='both')
    plt.tight_layout()
    if figpath:
        plt.savefig(figpath, dpi=dpi, bbox_inches='tight')
    plt.show()


# Plot sum of entropies for a given S_z value against chain size
# filepaths_and_sizes: list of (filepath, chain_size) tuples
# sz_values: single S_z value or list of S_z values to plot
# skips S_z values that don't exist in a file

def plot_entropy_sum_vs_chain_size(filepaths_and_sizes, sz_values, figsize=(8, 6), figpath=None, entropy='normalized', xlim=None, ylim=None, color=None):
    # filepaths_and_sizes: either a list of (filepath, chain_size, S) tuples (single dataset)
    #   or a dict {"label": [(filepath, chain_size, S), ...], ...} for multiple datasets
    # S is the spin value (1/2 or 1), used for normalization: n * log2(2S+1)
    # entropy: 'normalized' (col 1) or 'raw' (col 2)

    if not isinstance(sz_values, (list, np.ndarray)):
        sz_values = [sz_values]

    entropy_col = 1 if entropy == 'normalized' else 2

    # Normalize input: convert single list to dict with one entry
    if isinstance(filepaths_and_sizes, list):
        datasets = {"": filepaths_and_sizes}
    else:
        datasets = filepaths_and_sizes

    fig, ax = plt.subplots(figsize=figsize)

    # Collect all chain sizes across datasets for tick marks
    all_chain_sizes = set()

    total_lines = len(datasets) * len(sz_values)
    cmap = plt.colormaps.get_cmap('tab10').resampled(max(total_lines, 2))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']

    line_idx = 0
    for d_idx, (dataset_label, file_list) in enumerate(datasets.items()):
        marker = markers[d_idx % len(markers)]
        ls = '-'

        for sz_idx, sz_val in enumerate(sz_values):
            sizes = []
            entropy_sums = []
            for filepath, chain_size, S in file_list:
                try:
                    data = np.loadtxt(filepath, delimiter=',')
                except ValueError:
                    data = np.loadtxt(filepath)
                mask = np.isclose(data[:, 3], sz_val)
                if not np.any(mask):
                    continue

                all_chain_sizes.add(chain_size)
                entropy_sum = np.sum(data[mask, entropy_col])
                sizes.append(1/chain_size)
                entropy_sums.append(entropy_sum)

            if len(sizes) == 0:
                line_idx += 1
                continue

            if color is not None and total_lines == 1:
                c = color
            else:
                c = cmap(line_idx / max(total_lines - 1, 1))

            # Build label: combine dataset name and S_z
            if dataset_label:
                label = rf'{dataset_label}, $S_z = {sz_val:.0f}$'
            else:
                label = rf'$S_z = {sz_val:.0f}$'

            ax.scatter(sizes, entropy_sums, label=label, color=c, marker=marker, zorder=2)
            if len(sizes) >= 3:
                ax.plot(sizes, entropy_sums, ls, color=c, zorder=1)
            line_idx += 1

    all_chain_sizes = sorted(all_chain_sizes)
    tick_values = [1/L for L in all_chain_sizes]
    tick_labels = [r'$\frac{1}{' + str(L) + r'}$' for L in all_chain_sizes]

    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels)
    ax.set_yscale('symlog', linthresh=1e-1)

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(1, ax.get_ylim()[1] * 2)

    ax.set_xlabel('$1/L$')
    ax.set_ylabel(r'$\sum S$')
    ax.legend(ncol=2, fontsize='small', loc='best')
    ax.tick_params(direction='in', which='both')
    ax.grid(True, linestyle='--', alpha=0.25, zorder = 0)

    plt.tight_layout()
    if figpath:
        plt.savefig(figpath, dpi=500, bbox_inches='tight')
    plt.show()


def print_lowest_energies(filepaths, n=10, figsize=(8, 5), figpath=None, dpi=100, xlim=None, ylim=None):
    """Print and plot n lowest eigenvalues from multiple result files.

    filepaths: dict of {label: filepath} or list of (label, filepath) tuples.
    """
    if isinstance(filepaths, dict):
        filepaths = list(filepaths.items())

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    #cmap = plt.colormaps.get_cmap('tab10')

    for idx, (label, filepath) in enumerate(filepaths):
        try:
            data = np.loadtxt(filepath, delimiter=',')
        except ValueError:
            data = np.loadtxt(filepath)
        energies = np.sort(np.real(data[:, 0]))
        lowest = energies[:n]

        print(f"\n{label}  ({len(energies)} total states)")
        print(f"  {'i':>3}  {'E':>12}")
        print(f"  {'---':>3}  {'---':>12}")
        for i, e in enumerate(lowest):
            print(f"  {i:>3}  {e:>12.6f}")

        #color = cmap(idx / max(len(filepaths) - 1, 1))
        ax.scatter(range(len(lowest)), lowest, label=label, s=30, zorder=2) #color=color

    ax.set_xlabel('Index')
    ax.set_ylabel('Energy $E$')
    ax.set_title(f'{n} lowest eigenvalues')
    ax.legend(fontsize='small')
    ax.tick_params(direction='in', which='both')
    ax.grid(True, linestyle='--', alpha=0.25)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.tight_layout()
    if figpath:
        plt.savefig(figpath, dpi=dpi, bbox_inches='tight')
    plt.show()
    
def print_energies_momentum(filepath, figsize=(4.5,5.5), figpath=None, dpi=100):
    """Print and plot eigenvalues from momentum results file."""
    data = np.loadtxt(filepath, delimiter=',')
    energies = data[:, 0]
    sz_values = data[:, 1]
    k_values = data[:, 2]
    k_over_pi = data[:, 3]
    
    if len(energies) <= 16:
        print(f"\nMomentum eigenvalues ({len(energies)} total states)")
        print(f"  {'S_z':>6}  {'k/pi':>8}  {'E':>12}")
        print(f"  {'---':>6}  {'---':>8}  {'---':>12}")
        for sz, kp, e in zip(sz_values, k_over_pi, energies):
            print(f"  {sz:>6.1f}  {kp:>8.4f}  {e:>12.6f}")
    else: 
        print(f"\nMomentum eigenvalues ({len(energies)} total states)")
        print(f"  {'S_z':>6}  {'k/pi':>8}  {'E':>12}")
        print(f"  {'---':>6}  {'---':>8}  {'---':>12}")
        for sz, kp, e in zip(sz_values[0:10], k_over_pi[0:10], energies[0:10]):
            print(f"  {sz:>6.1f}  {kp:>8.4f}  {e:>12.6f}")

    # Spread degenerate points horizontally so each dot is visible
    k_plot = k_over_pi.copy()
    # Group by (k_rounded, E_rounded) and offset within each group
    k_round = np.round(k_over_pi, 6)
    e_round = np.round(energies, 6)
    spread = 0.06  # horizontal offset between degenerate dots
    visited = {}
    for i in range(len(k_plot)):
        key = (k_round[i], e_round[i])
        if key not in visited:
            visited[key] = []
        visited[key].append(i)
    for key, indices in visited.items():
        n = len(indices)
        if n > 1:
            offsets = np.linspace(-spread * (n - 1) / 2, spread * (n - 1) / 2, n)
            for j, idx in enumerate(indices):
                k_plot[idx] += offsets[j]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sc = ax.scatter(k_plot, energies, c=sz_values, cmap='viridis',
                    edgecolors='k', linewidths=0.3, s=40, zorder=2)
    cbar = plt.colorbar(sc, ax=ax)
    all_integer = np.all(np.isclose(sz_values, np.round(sz_values)))
    if all_integer:
        cbar.set_ticks(np.arange(sz_values.min(), sz_values.max() + 1))
    cbar.set_label(r'$S_z$')
    ax.set_xticks(np.unique(np.round(k_over_pi, 6)))
    ax.set_xlabel(r'$k (\pi)$')
    ax.set_ylabel(r'E (eV)')
    ax.tick_params(direction='in', which='both')
    ax.grid(True, linestyle='--', alpha=0.25)

    plt.tight_layout()
    if figpath:
        plt.savefig(figpath, dpi=dpi, bbox_inches='tight')
    plt.show()