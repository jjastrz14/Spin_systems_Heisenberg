#Writing to csv files and plotting results via matplotlib

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator, AutoLocator, LinearLocator, MaxNLocator
import os 

class Plots(object):
        
        def __init__(self, S, directory = None) -> None:
            
            #self.energies = energies 
            
            if S == 1:
                self.s_number = 's=1'
            elif S == 1/2:
                self.s_number = 's=1_2'
            
            if directory is not None:
                self.directory = os.path.join('./', directory)
                os.makedirs(directory, exist_ok=True)
            else:
                self.directory = './results_'+self.s_number
                
                
        def save_to_csv(self, x , name, header, real = False):
            
            if real == True: 
                np.savetxt(self.directory + str(name) + '.csv', np.transpose(np.real(x)), fmt='%.10f', delimiter=' ', header = header)
            else: 
                np.savetxt(self.directory + str(name) + '.csv', np.transpose(x), fmt='%.10f', delimiter=' ', header = header)

        def save_to_csv_without_transpose(self, x , name, header, real = False):
            
            if real == True: 
                np.savetxt(self.directory + str(name) + '.csv', np.real(x), fmt='%.10f', delimiter=' ', header = header)
            else: 
                np.savetxt(self.directory + str(name) + '.csv', x, fmt='%.10f', delimiter=' ', header = header)

        
        
        def plot_bands(self, energies, title, figsize, s, ticks, suffix):
            
            #Plotting energy bands with index
            #title -> header of plot, 
            #figsize -> size of figure, 
            #s -> thickness of a band 
            #ticks -> True for showing ticks on x axis 
            x = list(range(len(energies)))
            #energies_normalized = self.normalization_of_energies(self.energies)
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(x, energies, c = 'black', s=s, marker="_", linewidth=5, zorder=3)
            tick_spacing = 1
            if ticks == False: 
                ax.set_xticks([])
            else:
                ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_spacing))
            ax.grid(axis='y')
            ax.margins(0.1)
            ax.set_xlabel('index')
            ax.set_ylabel('E')
            ax.set_title(self.s_number + title)
            
            filename = 'bands_'
            if suffix is not None:
                filename += suffix
            plt.savefig(os.path.join(self.directory, filename + '.png'), bbox_inches='tight', dpi=200)
            plt.close()
            
            
        def plot_bands_with_s_z(self, energies, s_z_values, title, figsize, s, ticks, suffix):
        
            x = list(range(len(energies)))
            fig, ax = plt.subplots(figsize=figsize)
            
            ax.scatter(x, energies, c = 'black', s=s, marker="_", linewidth=5, zorder=3)
            
            for i, txt in enumerate(s_z_values):
                ax.annotate(txt, (x[i], energies[i]), xytext = (x[i] - 0.2, self.energies[i] + 0.05))
                ax.annotate("$S_z$", (x[i], energies[i]), xytext = (x[i] - 0.2, self.energies[i] - 0.07))
                
            tick_spacing = 1
            if ticks == False: 
                ax.set_xticks([])
            else:
                ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(tick_spacing))
            ax.grid(axis='y')
            ax.margins(0.1)
            ax.set_xlabel('index')
            ax.set_ylabel('E')
            ax.set_title(self.s_number + title)
            
            filename = 'bands_'
            if suffix is not None:
                filename += suffix
            plt.savefig(os.path.join(self.directory, filename + '.png'), bbox_inches='tight', dpi=200)
            plt.close()
            
        def plot_entropy(self, energies, entropy, color, title, figsize, s, suffix):
            # plotting - entropia dla odpowiedniej eigenenergii H 
            #entropy = self.normalization_of_entropy(entropy)
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(energies, entropy, c = color , s=s , marker="_", linewidth=5, zorder=3)
            ax.grid(axis='y')
            ax.margins(0.1)
            #start, end = ax.get_ylim()
            #ax.yaxis.set_ticks(np.arange(min(entropy), max(entropy)+1, 0.5))
            #ax.yaxis.set_major_locator(MaxNLocator())
            #ax.yaxis.set_minor_locator(AutoMinorLocator())
            
            ax.set_xlabel('Energy')
            ax.set_ylabel('Entropy')
            ax.set_title(self.s_number + title)
            
            '''
            ax.text(0.5, 0.5, '$S_z$ = '+str(S_z_total_number),
            horizontalalignment='center',
            verticalalignment='center',
            transform = ax.transAxes)
            '''
            
            filename = 'entropy_'
            if suffix is not None:
                filename += suffix
            plt.savefig(os.path.join(self.directory, filename + '.png'), bbox_inches='tight', dpi=200)
            plt.close()
            
        def plot_lambdas_entropy(self, lambdas, color, title, figsize, s, suffix):
            # plotting -lambdy dla odpowiedniej entropii
            
            fig, ax = plt.subplots(figsize=figsize)
            for i in range(len(lambdas)):
                l1= ax.scatter([i]*len(lambdas[i]),lambdas[i], c = color , s=s , marker="_", linewidth=5, zorder=3)
                l2 =ax.scatter(i,sum(lambdas[i]), c = "green" , s=s , marker="_", alpha=.5, linewidth=4, zorder=3)
            
            ax.grid(axis='y')
            start, end = ax.get_ylim()
            ax.yaxis.set_ticks(np.arange(start, end, 0.05))
            ax.margins(0.1)
            ax.set_ylim(bottom=-0.04, top = 1.04)
            ax.set_xlabel('Number of eigenvector')
            ax.set_ylabel('energy of $\lambda$')
            ax.set_title(self.s_number + title)
            ax.legend((l1, l2), ('$\lambda$', 'Sum of $\lambda$'), loc='upper left', shadow=False)
            
            filename = 'lambda_entropy'
            if suffix is not None:
                filename += suffix
            plt.savefig(os.path.join(self.directory, filename + '.png'), bbox_inches='tight', dpi=200)
            plt.close()
            
        def plot_s_z(self,s_z_values, energies, color, title, figsize, s, suffix):
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.scatter(s_z_values, energies, c = color , s=s , marker="_", alpha = .8, linewidth=8, zorder=3)
            
            ax.grid(axis='y')
            ax.margins(0.1)
            ax.set_xticks(s_z_values)
            ax.set_xlabel('$S_z$')
            ax.set_ylabel('$E$')
            ax.set_title(self.s_number + title)
            
            filename = 's_z_energy'
            if suffix is not None:
                filename += suffix
            plt.savefig(os.path.join(self.directory, filename + '.png'), bbox_inches='tight', dpi=200)
            plt.close()