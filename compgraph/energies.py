def StochasticEnergy(random_config_batch, graph_tuples_batch, ansatz, Hamiltonian):
    factor_list=[]
    ansatz_list=[]
    configurations=random_config_batch
    for idx, graph_tuple in enumerate(graph_tuples_batch):
        ansatz_amplitude, ansatz_phase = ansatz(graph_tuple)[0]
        amplitude=tf.cast(ansatz_amplitude, tf.float32)
        ansatz_list.append(amplitude)
        
        phase=tf.cast(ansatz_phase, tf.float32)
        complex_coefficient = tf.complex(amplitude, phase)

        factor_list.append(complex_coefficient)
        print("index", idx, "amplitude", amplitude, configurations[idx])
    energy=0
    #wave_function_csr = compute_wave_function_csr(graph_tuples_batch, ansatz, random_config_batch)
    #print("Confronting the wave fucntion", wave_function_csr, "with the factor list \n" , factor_list,)
    for i in range(len(configurations)):

        v2= configurations[i].transpose()
        v2_factor= factor_list[i]
        en=0
        for index in range(len(configurations)):
            v1= configurations[index]
            v1_factor= factor_list[index]
            psi2=v2_factor*v2
            psi1=v1_factor*v1
            value = psi1.conj().dot(Hamiltonian.dot(psi2))
            #print("this is value", value,"\n","this is v2", v2, "\n", "this is v1", v1, "\n")
            en+=value
        energy += en    
        
    return energy


def Overlap(random_config_batch, graph_tuples_batch, wave1, wave2):
    amplitude1_list= []
    phase1_list=[]
    amplitude2_list= []
    phase2_list=[]
    configurations=random_config_batch
    for graph in range(len(graph_tuples_batch)):
        ansatz1_amplitudes, ansatz1_phases = wave1(graph_tuples_batch[graph])[0]
        amplitude1_list.append(ansatz1_amplitudes)
        phase1_list.append(ansatz1_phases)
        ansatz2_amplitudes, ansatz2_phases = wave2(graph_tuples_batch[graph])[0]
        amplitude2_list.append(ansatz2_amplitudes)
        phase2_list.append(ansatz2_phases)   
    pass   