# MaxCutQuiskitSeminar
Requirements Versions of Python Packages
qiskit 1.2.4
numpy 2.1.3
network 3.4.2
scipy 1.14.1

https://github.com/rsln-s/IEEE_QW_2020/
https://en.wikipedia.org/wiki/BQP
https://en.wikipedia.org/wiki/Quantum_optimization_algorithms

How to use:
Simple run.

to set your own graphs you can define those in obtain_graphs(start, end) function, 
that returns an array of graphs from start till end both inclusive

Then after all function definition in line 288 - 291 you have a classical solver for MaxCut, 
that simple call a function solve_and_draw_graph(graph, Solve) where you pass your nx Graph and boolean to solve it or simple to draw
solution will be saved here f"solutionAndTime/graph{G.number_of_nodes()}.txt" and image of graph: f"graphs/graph{G.number_of_nodes()}.png"

Second part is from 293 to 312 line. It is quantum solver. Here you can define also graphs, number of shots, starting p, and init_params for p
then it will iteratively go through each graph in array with incrementing p until 20 and also messes the time 
and saves it to f"QuantumSolutionAndTime/graph{G.number_of_nodes()}.txt". It uses quantum_solve_and_print(G, p, params, shots = 1024) that takes 
Graph, number p, thetas for p, and optimal number of shots

The last part is from 315 to 356, it iterates over graphs indexes and summing the probabilities of the best solutions. It gets solutions from above defined path in first part 
and probabilities it takes from file from second part. 
Also it draws graphs comparing two runs it saves probability to probabilities folder and took old from folder "firstRun"  
get_probability_for_firstRun(f"firstRun/probabilities/graph{i}.txt")
this function return an array of the probabilities from firstRun