import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import matplotlib.cm as cm


# function to plot Entropy against eigenvalue with cmap of Sz
# mode='leq' filters S_z <= sz_filter, mode='eq' filters S_z == sz_filter
def plot_entropy_against_eigenvalue(filepath, figsize=(8, 6), sz_filter=0, mode='leq', figpath=None, entropy='normalized'):
    # entropy: 'normalized' (col 1) or 'raw' (col 2)
    data = np.loadtxt(filepath)
    energies = data[:, 0]
    entropy_col = 1 if entropy == 'normalized' else 2
    entropies = data[:, entropy_col]
    sz = data[:, 3]

    # store full S_z range for consistent color normalization
    sz_min, sz_max = sz.min(), sz.max()

    if mode == 'eq':
        mask = np.isclose(sz, sz_filter)
    else:
        mask = sz <= sz_filter
    energies = energies[mask]
    entropies = entropies[mask]
    sz = sz[mask]

    # use integer S_z if all values are whole numbers
    all_integer = np.all(np.isclose(sz, np.round(sz)))
    if all_integer:
        sz = np.round(sz).astype(int)

    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(energies, entropies, c=sz, cmap='viridis', edgecolors='k', linewidths=0.01,
                    vmin=sz_min, vmax=sz_max)
    if mode != 'eq':
        cbar = plt.colorbar(sc, ax=ax)
        if all_integer:
            cbar.set_ticks(np.arange(sz.min(), sz.max() + 1))
        cbar.set_label(r'$S_z$')
    ax.set_xlabel(r'Eigenvalue $E$')
    ax.set_ylabel(r'Entropy $S$')
    ax.tick_params(direction='in', which='both')
    plt.tight_layout()
    if figpath:
        plt.savefig(figpath)
    plt.show()
    
    
# Plot sum of entropies for a given S_z value against chain size
# filepaths_and_sizes: list of (filepath, chain_size) tuples
# sz_values: single S_z value or list of S_z values to plot
# skips S_z values that don't exist in a file

def plot_entropy_sum_vs_chain_size(filepaths_and_sizes, sz_values, figsize=(8, 6), figpath=None, entropy='normalized', xlim=None, ylim=None):
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

    cmap = plt.colormaps.get_cmap('coolwarm').resampled(max(len(sz_values), 2))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    linestyles = ['-', '--', '-.', ':']

    for d_idx, (dataset_label, file_list) in enumerate(datasets.items()):
        marker = markers[d_idx % len(markers)]
        ls = linestyles[d_idx % len(linestyles)]

        for sz_idx, sz_val in enumerate(sz_values):
            sizes = []
            entropy_sums = []
            for filepath, chain_size, S in file_list:
                data = np.loadtxt(filepath)
                mask = np.isclose(data[:, 3], sz_val)
                if not np.any(mask):
                    continue

                all_chain_sizes.add(chain_size)
                d = int(2*S + 1)  # local dimension: 2 for S=1/2, 3 for S=1
                n = chain_size // 2  # subsystem size
                max_entropy = n * np.log2(d)
                entropy_sum = np.sum(data[mask, entropy_col]) / max_entropy if max_entropy > 0 else 0
                sizes.append(1/chain_size)
                entropy_sums.append(entropy_sum)

            if len(sizes) == 0:
                continue

            color = cmap(sz_idx)

            # Build label: combine dataset name and S_z
            if dataset_label:
                label = rf'{dataset_label}, $S_z = {sz_val:.0f}$'
            else:
                label = rf'$S_z = {sz_val:.0f}$'

            ax.scatter(sizes, entropy_sums, label=label, color=color, marker=marker, zorder=2)
            if len(sizes) >= 3:
                ax.plot(sizes, entropy_sums, ls, color=color, zorder=1)

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
    ax.legend(ncol=2, fontsize='small', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.tick_params(direction='in', which='both')
    ax.grid(True, linestyle='--', alpha=0.25, zorder = 0)

    plt.tight_layout()
    if figpath:
        plt.savefig(figpath, dpi=500, bbox_inches='tight')
    plt.show()