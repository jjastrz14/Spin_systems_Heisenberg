# version 3.0 for spin 1D chain diagonalization (max number of sites 8-10) using numpy and:
# -> calculating a density matrix
# -> calculating a reduced density matrix of chain divided into two equal subsytems
# -> calculating entropy of this subsystems
# -> making a matrix block diagonal by simple matrix operations
# -> calculation of rho reduced density matrices with fixed s_z quantum number
####


import numpy as np
import Heisenberg_chain
import Plotting_writing
from itertools import groupby
from collections import Counter
import argparse
import math

TRACE_TOLERANCE = 1e-12



if __name__ == '__main__':

    ######## Heisenberg Graph #########
    '''
    Program for calculating and diagonalization of Heisenberg Hamiltonian for graph with defined adjacency matrix                               #
    # class Graph - define graph, class Heisenberg - calculation of H
    # here you can creating a graph using add_edge methods or declare prepared matrix (adjMatrix).                                              #
    # N is a size of the system #
    '''

    parser = argparse.ArgumentParser(description="Heisenberg / AKLT spin chain diagonalization")
    parser.add_argument("-N", "--sites", type=int, default=8, help="Number of sites in the chain")
    parser.add_argument("-S", "--spin", type=float, default=0.5, help="Spin value: 0.5 or 1")
    parser.add_argument("-b", "--boundary", type=str, default="PBC", choices=["PBC", "OBC"],
                        help="Boundary conditions: PBC (periodic) or OBC (open)")
    parser.add_argument("-m", "--model", type=str, default="Heisenberg", choices=["Heisenberg", "AKLT"],
                        help="Model: Heisenberg or AKLT")
    args = parser.parse_args()

    size_of_the_chain = args.sites
    boundary = args.boundary
    S = args.spin
    model = args.model

    adjMatrix = np.eye(size_of_the_chain, k=1, dtype=int)
    if boundary == "PBC":
        adjMatrix[-1][0] = 1
    verbose = size_of_the_chain <= 6
    print(f"Boundary conditions: {boundary}")
    if verbose:
        print("This is adjacency Matrix : \n", adjMatrix)
    
    N = len(adjMatrix) #size of the system
    print("Calculating N = " + str(N)  + " system for S = " + str(S))
    
    H = Heisenberg_chain.Heisenberg(N, S)
    print("Start the diagonalization")
    
    basis_H, basis_H_s_z, spins = H.calculate_basis()

    if verbose:
        print("Basis S_z: ", basis_H_s_z)
    print("List of possible values of S_z: ", spins)
    print("Dictionary of spin z occurances: ", Counter(basis_H_s_z).values() )
    if model == "AKLT":
        H.create_Hamiltonian_AKLT(adjMatrix)
    else:
        H.create_Hamiltonian(adjMatrix)
    
    if verbose:
        print(H.H)
    all_energies = []
    entropy_all_system = []
    entropy_raw_all = []
    eigen_rho_sys_all = []
    s_z_number = []
    s_z_lambdas = []
    sum_lambdas = []
    psi_shape = []
    
    # For odd N: A gets the extra site (A >= B)
    size_of_sub_A = math.ceil(N / 2)
    size_of_sub_B = N - size_of_sub_A
    
    print(f"This is size A {size_of_sub_A}")
    print(f"This is size B {size_of_sub_B}")
    
    #size_of_sub_A = 3
    #size_of_sub_B = 1
    #spins = [0.0]
    for i , spin in enumerate(spins):
        energies, vectors, spin_basis = H.block_Hamiltonian(i)
        
        if verbose:
            print(f"S_z = {spins[i]} start")
        #print(f"Spin basis: {spin_basis}")
        
        #calculation of new basis
        subsystem_A, subsystem_B, new_basis = H.subsystems_fixed_s_z(spin_basis,size_of_sub_A,size_of_sub_B)
        
        sum_entropy = 0
        for j in range(len(energies)):
            
            #print(f"Eigenvectors for this spin: {vectors[:,j]}")
            #print(set(vectors[:,j]))
            
            #if len(set(np.around(vectors[:,j],decimals = 5))) == 1:
                #print((f"Eigenvectors with equal elements: {vectors[:,j]}"))
            #print(f"This is energy {energies[j]}")   
            #print(f"This is vectors {vectors[:,j]}")

            psi = np.zeros(shape=(len(subsystem_A),len(subsystem_B)), dtype = complex)
            for k,v in enumerate(vectors[:,j]):
                #print(f"This is {v}")
                psi[new_basis[k][0]][new_basis[k][1]] = v #0 and 1 becasue it's a matrix 
                #print("This is value of psi ", v)
            
            psi_shape.append(psi.shape)
            
            
            #print(f"This is psi vector {j}: \n", psi)
            #subsystemA
            rho = np.dot(psi, psi.conj().transpose()) 
            #print(f"This len is rho vector {j}: \n", len(rho))
            
            #trace calculation
            if abs(np.trace(rho) - 1.0) > TRACE_TOLERANCE:
                    print("Trace of the system: ", np.trace(rho))
            
            #entropy
            entropy_sys, entropy_raw, eigen_rho_sys = H.calculate_entropy(rho, size_of_sub_A)
            #print(f"This is entropy of j-th {j} energy {energies[j]} with spin S_z {spins[i]} : {entropy_sys}")
            sum_entropy += entropy_sys

            #print(f"Lambdas : {eigen_rho_sys}")

            entropy_all_system.append(entropy_sys)
            entropy_raw_all.append(entropy_raw)
            s_z_number.append(spins[i])
            
            s_z_lambdas.append([spins[i]]*len(eigen_rho_sys))
            #for lambdas from reduced density matrices
            eigen_rho_sys_all.append(eigen_rho_sys)
        
        psi_shape.append(psi.shape)    
        #print(f"This is sum of this S_z {spins[i]} entropies {sum_entropy}")

        all_energies.append(energies)

    for k in range(len(eigen_rho_sys_all)):
            sum_lambdas.append(sum(eigen_rho_sys_all[k]))

    if verbose:
        print(psi_shape)
        
    #print(sum_lambdas)
                
    all_energies = np.concatenate(all_energies)
    #energies_sorted = np.sort(all_energies)
    #print("Energies: ", energies_sorted)
    
    print("Start writing to files")
    if S == 1/2:
        S_save = "1_2"
    else:
        S_save = "1"
        
    
    #class initialization
    Plotting = Plotting_writing.Plots(S, directory = "./results/results_" + model + "_" + str(len(adjMatrix)) + "_sites_" + S_save + "_" + boundary)
        
    results = [all_energies, entropy_all_system, entropy_raw_all, s_z_number, sum_lambdas]
    Plotting.save_to_csv(results, name = "/results_" + S_save + "_" + str(N),
                         header = "Energy Entropy_normalized Entropy_raw S_z Sum_lambdas", real = True)

    #lambdas = np.column_stack([np.concatenate(eigen_rho_sys_all, axis = 0) , np.concatenate(s_z_lambdas)])
    #Plotting.save_to_csv_without_transpose(lambdas, name = "/lambdas_" + S_save + "_" + str(N), header = "Lambdas S_z", real = True)
    print("Writing to files done")
    
    
    #Plotting.plot_lambdas_entropy(eigen_rho_sys_all, color = 'black', title = ", lambda, " + str(N) +" sites, sys ", figsize=(10,12),s=400, suffix = str(N) + "_sys_lambda")
    #Plotting.plot_entropy(entropy_all_system, color = 'red', title = ", " + str(N) +" sites, system ", figsize=(10,12),s=550, suffix = str(N) + "_sys_entropy")
    
    print("Start plotting")
    
    if len(adjMatrix) <= 4: 
        ticks = True 
    else: ticks = False
    
    if N <= 8: 
        #plots of energy bands 
        Plotting.plot_bands(np.sort(all_energies), title = ", " +str(N) +" sites, graph", figsize=(10,12),s=550, ticks = ticks, suffix = str(N) +"_sites_chain")
        #Plotting.plot_bands_with_s_z(np.around(S_z_total,0), title = ", " + str(N+1) +" sites, graph", figsize=(10,12),s=550, ticks = True, suffix = str(N+1) +"S_z_sites_chain")
        Plotting.plot_s_z(sorted(basis_H_s_z), all_energies, color = 'dodgerblue', title = ", " + str(N) +" sites, graph", figsize=(10,12),s=550, suffix = str(N) +"_sites_Sz")
        Plotting.plot_entropy(all_energies, entropy_all_system, color = 'red', title = ", " + str(N) +" sites, system ", figsize=(10,12),s=550, suffix = str(N) + "_sys_entropy")
        #Plotting.plot_entropy(entropy_all_env, color = 'blue', title = ", " + str(N+1) +" sites, environment ", figsize=(10,12),s=550, suffix = str(N+1) + "_env_entropy")
        #Plotting.plot_lambdas_entropy(eigen_rho_env_all, color = 'black', title = ", lambda, " + str(N+1) +" sites, env ", figsize=(10,12),s=400, suffix = str(N+1) + "_env_lambda")
        Plotting.plot_lambdas_entropy(eigen_rho_sys_all, color = 'black', title = ", lambda, " + str(N) +" sites, sys ", figsize=(10,12),s=400, suffix = str(N) + "_sys_lambda")
        print("Plotting of bands done") 

    
    print("Success")





    
    
    
    

