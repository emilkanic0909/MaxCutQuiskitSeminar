from qiskit.primitives import Sampler
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import numpy as np
import networkx as nx
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time

#start and end inclusive
def obtain_graphs(start = 2, end = 20):
    G2 = nx.Graph()
    G3 = nx.Graph()
    G4 = nx.Graph()
    G5 = nx.Graph()
    G6 = nx.Graph()
    G7 = nx.Graph()
    G8 = nx.Graph()
    G9 = nx.Graph()
    G10 = nx.Graph()
    G11 = nx.Graph()
    G12 = nx.Graph()
    G13 = nx.Graph()
    G14 = nx.Graph()
    G15 = nx.Graph()
    G16 = nx.Graph()
    G17 = nx.Graph()
    G18 = nx.Graph()
    G19 = nx.Graph()
    G20 = nx.Graph()

    # Graph with 2 vertices: A single edge connecting the two vertices
    G2.add_edges_from([[0, 1]])

    # Graph with 3 vertices: A complete graph (triangle)
    G3.add_edges_from([[0, 1], [0, 2], [1, 2]])

    # Graph with 4 vertices: A cycle with an additional diagonal edge
    G4.add_edges_from([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])

    # Graph with 5 vertices: A bipartite graph connecting vertices 0-2 to vertices 3-4
    G5.add_edges_from([[0, 3], [0, 4], [1, 3], [1, 4], [2, 3], [2, 4]])

    # Graph with 6 vertices: A cycle with chords connecting opposite vertices
    G6.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0],
        [0, 3], [2, 5]
    ])

    # Graph with 7 vertices: A cycle with additional chords for increased connectivity
    G7.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 0],
        [0, 3], [1, 4], [2, 5]
    ])

    # Graph with 8 vertices: A cube structure represented in 3D graph form
    G8.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 0],  # Base square
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top square
        [0, 4], [1, 5], [2, 6], [3, 7]  # Vertical connections
    ])

    # Graph with 9 vertices: A 3x3 grid graph representing a mesh network
    G9.add_edges_from([
        [0, 1], [1, 2],
        [3, 4], [4, 5],
        [6, 7], [7, 8],
        [0, 3], [3, 6],
        [1, 4], [4, 7],
        [2, 5], [5, 8]
    ])

    # Graph with 10 vertices: A cycle with cross connections for added complexity
    G10.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
        [5, 6], [6, 7], [7, 8], [8, 9], [9, 0],
        [0, 5], [2, 7]
    ])

    # Graph with 11 vertices: A cycle with chords to increase edge density
    G11.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5],
        [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 0],
        [0, 5], [2, 7], [4, 9]
    ])

    # Graph with 12 vertices: A cycle with opposite vertices connected
    G12.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 0],
        [0, 6], [3, 9]
    ])

    # Graph with 13 vertices: A cycle with chords every third vertex
    G13.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 0],
        [0, 6], [3, 9], [6, 12]
    ])

    # Graph with 14 vertices: A large cycle with evenly spaced chords
    G14.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 0],
        [0, 7], [3, 10], [6, 13]
    ])

    # Graph with 15 vertices: A cycle with chords connecting vertices at intervals
    G15.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 0],
        [0, 7], [3, 10], [6, 13]
    ])

    # Graph with 16 vertices: A cycle with diametrically opposite connections
    G16.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
        [12, 13], [13, 14], [14, 15], [15, 0],
        [0, 8], [4, 12]
    ])

    # Graph with 17 vertices: A cycle with chords connecting every eighth vertex
    G17.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
        [12, 13], [13, 14], [14, 15], [15, 16], [16, 0],
        [0, 8], [4, 12], [8, 16]
    ])

    # Graph with 18 vertices: A cycle with chords connecting every ninth vertex
    G18.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
        [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 0],
        [0, 9], [4, 13], [8, 17]
    ])

    # Graph with 19 vertices: A cycle with chords connecting vertices at regular intervals
    G19.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
        [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 0],
        [0, 9], [4, 13], [8, 17]
    ])

    # Graph with 20 vertices: A cycle with cross connections at opposite vertices
    G20.add_edges_from([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6],
        [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12],
        [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 0],
        [0, 10], [5, 15]
    ])
    graphs = [G2, G3, G4, G5, G6, G7, G8, G9, G10, G11, G12, G13, G14, G15, G16, G17, G18, G19, G20]
    return graphs[(start - 2):end + 1]

def get_qaoa_circuit(G, beta, gamma):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    qc.h(range(N))
    for i in range(len(beta)):
        qc = qc.compose(get_cost(G, gamma[i]))
        qc = qc.compose(get_mixer(G, beta[i]))
    qc.barrier(range(N))
    qc.measure(range(N), range(N))#this is expectation value from Hamiltonian woth respect to Pauli-Z
    return qc

#cost operator should give more hardness to schema if it more close to solution. Level of hardness is depends on GAMMA parameter
def get_cost(G, gamma):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for q1,q2 in G.edges():
        qc.cx(q1, q2)
        qc.rz(2 * gamma, q2)
        qc.cx(q1, q2)
    return qc

#mixed operator should make more randomnes to get other scheme if solution is far. Level of mixing represented through BETA parameter
def get_mixer(G, beta):
    N = G.number_of_nodes()
    qc = QuantumCircuit(N,N)
    for q1 in G.nodes():
        qc.rx(2 * beta, q1)
    return qc

def invert_counts(counts):
    return {k[::-1]:v for k,v in counts.items()}

def maxcut_obj(x, G):
    cut = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            cut -=1
    return cut

def compute_energy(counts, G):
    energy = 0
    total_counts = 0
    for meas, meas_count in counts.items():
        obj_for_meas = maxcut_obj(meas, G)
        energy += obj_for_meas * meas_count
        total_counts += meas_count
    return energy / total_counts

def get_objective_function(G,p, shots = 128):
    def f(theta):
        sampler = Sampler(options={'shots': shots})
        job = sampler.run(get_qaoa_circuit(G, theta[:p], theta[p:]))
        inverted_counts = invert_counts(job.result().quasi_dists[0].binary_probabilities())
        return compute_energy(inverted_counts, G)
    return f

def classical_MaxCutSolver(G):
    numberOfNodes = G.number_of_nodes()
    currentMaxString = []
    currentMaxNumber = 0
    for i in range(1, (2**numberOfNodes) - 1):
        first_set = [j for j, bit in enumerate(bin(i)[2:].zfill(numberOfNodes)) if bit == '0']
        second_set = [j for j, bit in enumerate(bin(i)[2:].zfill(numberOfNodes)) if bit == '1']
        tempNumber = 0
        for j in first_set:
            for n in second_set:
                if G.has_edge(j,n):
                    tempNumber+=1
        if currentMaxNumber==tempNumber:
            currentMaxString.append(bin(i)[2:].zfill(numberOfNodes))
        elif currentMaxNumber < tempNumber:
            currentMaxNumber=tempNumber
            currentMaxString.clear()
            currentMaxString.append(bin(i)[2:].zfill(numberOfNodes))
    return currentMaxString


def solve_and_draw_graph(G, Solve = True):
    #pos = nx.spring_layout(G, seed=42, k=0.5)
    pos = nx.kamada_kawai_layout(G)
    plt.clf()
    nx.draw(G, pos,
            with_labels=True,
            node_size=500,
            node_color="skyblue",
            edge_color="black",
            font_size=25,
            font_color="black")
    plt.savefig(f"graphs/graph{G.number_of_nodes()}.png")
    plt.clf()
    if Solve:
        start_time = time.time()
        classic = classical_MaxCutSolver(G)
        end_time = time.time()
        with open(f"solutionAndTime/graph{G.number_of_nodes()}.txt", "w", encoding="utf-8") as file:
            file.write(f"{classic}\n")
            file.write(f"workingtime: {end_time - start_time:.5f} seconds")
        print(f"solved and drawed graph{G.number_of_nodes()}")
    return

def quantum_solve_and_print(G, p, params, shots = 1024):
    start_time = time.time()

    optimization_start_time = time.time()
    func = get_objective_function(G, p, shots)
    res_sample = minimize(func, params, method='COBYLA',options={'maxiter': 4000, 'disp': False})
    optimization_end_time = time.time()
    sampler = Sampler(options={'shots': shots})
    optimal_theta = res_sample['x']
    binary_probabilities = invert_counts(
        sampler.run(get_qaoa_circuit(G, optimal_theta[:p], optimal_theta[p:])).result().quasi_dists[
            0].binary_probabilities())

    end_time = time.time()
    with open(f"QuantumSolutionAndTime/graph{G.number_of_nodes()}.txt", "a", encoding="utf-8") as file:
        file.write(f"with p = {p}, shots = {shots}, optimization time: {optimization_end_time - optimization_start_time:.5f} second, total workingtime: {end_time - start_time:.5f} seconds, with the biggest 100 probabilities:\n")
        for key, value in sorted(binary_probabilities.items(), key=lambda item: item[1])[-100:]:
            file.write(f"'{key}': {value} \n")
        file.write("\n\n")
    print(f"solved and drawed graph{G.number_of_nodes()} with p = {p}")
    return res_sample

def get_probability_for_firstRun(filePath):
    probabilities = []
    with open(filePath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if "by p =" in line and "is probability of correct solution =" in line:
                parts = line.split("is probability of correct solution =")
                probability = float(parts[1].strip())
                probabilities.append(probability)
    return probabilities
#To solve all graphs with classical solver
#graphArray = obtain_graphs(2,20)
#for graph in graphArray:
 #   solve_and_draw_graph(graph, True)

#to solve each graph with incrementing p
#graphNumber = 2   #from which graph to start
#graphs = obtain_graphs(2,19)   #get graphs
#shots = 2048   #number of shots
#p = 1  #optimization number
#init_params = np.random.rand(2 * p)
#G = graphs[graphNumber - 2]
#time_to_cut = time.time()
#while True:
#    if (p == 20) or (time.time() - time_to_cut > 900):
#        p = 1
#        init_params = np.random.rand(2 * p)
#        graphNumber += 1
#        G = graphs[graphNumber - 2]
#        time_to_cut = time.time()
#    if (graphNumber == 19 and p >= 4):
#        break
#    result = quantum_solve_and_print(G, p,init_params, shots)
#    p+=1
#    init_params = np.concatenate([result.x, np.random.rand(1), np.random.rand(1)])

#calculate possiibilities and maybe draw
for i in range(2,20):
    xDots = []
    yDots = []
    yDotsFirstRun = get_probability_for_firstRun(f"firstRun/probabilities/graph{i}.txt")
    with open(f"solutionAndTime/graph{i}.txt", 'r') as file:
        lines = file.readlines()
        binary_strings = []
        for line in lines:
            line = line.strip().strip("[]").replace("'", "").split(',')
            for element in line:
                element = element.strip()
                if all(char in '01' for char in element):
                    binary_strings.append(element)
    print(binary_strings)
    with open(f"QuantumSolutionAndTime/graph{i}.txt", 'r') as file2:
        content = file2.read()
        blocks = content.strip().split('\n\n\n')
        for block in blocks:
            probability = 0.0
            lines = block.split('\n')
            if 'p =' in lines[0]:
                p_value = int(lines[0].split('p =')[1].split(',')[0].strip())
            probabilities = {}
            for line in lines:
                if ":" in line and "'" in line:
                    key, value = line.strip().replace("'", "").split(':')
                    probabilities[key.strip()] = float(value.strip())
            for string in binary_strings:
                if string in probabilities:
                    probability += probabilities[string]

            with open(f'probabilities/graph{i}.txt', "a") as write_file:
                #write_file.write(f"by p = {p_value} is probability of correct solution = {probability} \n");

                xDots.append(p_value)
                yDots.append(probability)
        plt.clf()
        plt.plot( xDots, yDots, color='blue')
        plt.plot(xDots, yDotsFirstRun, color='red')
        plt.ylim(0, 1.1000)
        plt.savefig(f"comparingGraphFirstAndSecondRun/graph{i}.png")
        plt.clf()