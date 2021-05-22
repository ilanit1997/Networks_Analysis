from typing import List, Set, Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx
import copy
import random



def LTM(graph: networkx.Graph, patients_0: List, iterations: int) -> Set:
    It = set(patients_0)
    St = set(graph.nodes) - It
    for n in St:
        graph.nodes[n]['concern'] = 0
    for i in range(iterations):
        NIt = set()
        for n in St:
            neightboors = set(graph.adj[n])
            infected_neig = neightboors & It
            infected_weights = sum(graph[n][ind]['w'] for ind in infected_neig)
            isInfected = True if CONTAGION*infected_weights>= 1 + graph.nodes[n]['concern'] else False
            if isInfected:
                NIt.add(n)
        St = St - NIt
        for n in St:
            neightboors = set(graph.adj[n])
            graph.nodes[n]['concern'] = len(neightboors & It) / len(neightboors)
        It = It.union(NIt)
    return It


def ICM(graph: networkx.Graph, patients_0: List, iterations: int) -> [Set, Set]:
    total_infected = set(patients_0)
    r = np.random.binomial(n=1, p=LETHALITY, size=len(total_infected))
    total_deceased = set(x for i, x in enumerate(list(total_infected)) if r[i] == 1)
    total_infected -= total_deceased
    It = set()
    It_1 = set(x for x in total_infected)
    infected_deceased = total_infected.union(total_deceased)
    St = set(graph.nodes) - (infected_deceased)
    Rt_1 = set(x for x in total_deceased)
    Rt = set()
    NIt = set()
    NIt_1 = set(x for x in total_infected)
    for n in St:
        graph.nodes[n]['concern'] = 0
    for i in range(iterations):
        for v in St:
            neightboors = set(graph.adj[v]) & NIt_1
            for u in neightboors:
                pt = min(1, CONTAGION * graph[v][u]['w'] * (1 - graph.nodes[v]['concern']))
                pI = np.random.binomial(n=1, p=pt)
                if pI == 1:
                    pD = np.random.binomial(n=1, p=LETHALITY)
                    if pD == 1:
                        Rt.add(v)
                    else:
                        NIt.add(v)
                    break
        Rt = Rt.union(Rt_1)
        It = It.union(It_1).union(NIt) - Rt
        St = set(graph.nodes) - (It.union(Rt))
        for v in St:
            cur = set(graph[v])
            if len(cur) == 0:
                temp = 1
            else:
                temp = (len(cur & It_1) + 3 * len(cur & Rt_1)) / len(cur)
            graph.nodes[v]['concern'] = min(1, temp)
        It_1 = set(e for e in It)
        NIt_1 = set(e for e in NIt)
        NIt = set()
        Rt_1 = set(e for e in Rt)
    return [It, Rt_1]


def plot_degree_histogram(histogram: Dict):
    plt.bar(*zip(*histogram.items()))
    plt.xlim(0, max(histogram.keys()))
    plt.show()


def calc_degree_histogram(graph: networkx.Graph) -> Dict:
    """
    Example:
    if histogram[1] = 10 -> 10 nodes have only 1 friend
    """
    histogram = {}
    for node in list(graph.nodes):
        if graph.degree[node] not in histogram.keys():
            histogram[graph.degree[node]] = 1
        else:
            histogram[graph.degree[node]] +=1
    return histogram


def build_graph(filename: str) -> networkx.Graph:
    data = pd.read_csv(filename)
    if 'w' in data.columns:
        data1 = data[['from', 'to', 'w']]
        G = networkx.from_pandas_edgelist(data1, 'from', 'to', 'w')
    else:
        data1 = data[['from', 'to']]
        G = networkx.from_pandas_edgelist(data1, 'from', 'to')
    return G


def clustering_coefficient(graph: networkx.Graph) -> float:
    adj_matrix = networkx.convert_matrix.to_numpy_matrix(graph)
    adj_matrix[adj_matrix > 0] = int(1)
    two_edges_trip = adj_matrix @ adj_matrix
    three_edges_trip = two_edges_trip @ adj_matrix
    connect_trip = np.sum(two_edges_trip) - np.sum(np.trace(two_edges_trip))
    triangles = np.sum(np.trace(three_edges_trip))
    return triangles / connect_trip



def compute_lethality_effect(graph: networkx.Graph, t: int) -> [Dict, Dict]:
    global LETHALITY
    mean_deaths = {}
    mean_infected = {}
    for l in (.05, .15, .3, .5, .7):
        LETHALITY = l
        r1 = []
        r2 = []
        for iteration in range(30):
            G = copy.deepcopy(graph)
            patients_0 = np.random.choice(list(G.nodes), size=50, replace=False, p=None)
            res = ICM(G,patients_0,t)
            r1.append(res[0])
            r2.append(res[1])
        mean_deaths[l] = np.mean(r2)
        mean_infected[l] = np.mean(r1)
    return mean_deaths, mean_infected


def plot_lethality_effect(mean_deaths: Dict, mean_infected: Dict):
    plt.plot(mean_deaths.keys(), mean_deaths.values(), 'red')
    plt.plot(mean_infected.keys(), mean_infected.values(), 'blue')
    plt.show()

#choose Select a list of influential nodes in a graph using VoteRank algorithm
def choose_who_to_vaccinate(graph: networkx.Graph) -> List:
    """
        The following heuristic for Part C is simply taking the top 50 friendly people;
         that is, it returns the top 50 nodes in the graph with the highest degree.
        """
    node2degree = dict(graph.degree)
    sorted_nodes = sorted(node2degree.items(), key=lambda item: item[1], reverse=True)[:50]
    people_to_vaccinate = [node[0] for node in sorted_nodes]
    return people_to_vaccinate


"Global Hyper-parameters"
CONTAGION = .8
LETHALITY = .15

if __name__ == "__main__":
    filenameA1 = 'PartA1.csv'
    filenameA2 = 'PartA2.csv'
    filenameBC = 'PartB-C.csv'
    random.seed(3)
    G1 = build_graph(filename=filenameA1)
    G2 = build_graph(filename=filenameA2)
    G3 = build_graph(filename=filenameBC)
    patients = pd.read_csv('patients0.csv', header=None)[0].tolist()
    sick = []
    dead = []
    for i in range(5):
        patients = random.sample(patients, 50)
        result = ICM(G3,patients , 6)
        print("original:")
        print(result)
        print("ours:")
        res2 = choose_who_to_vaccinate(G3)
        G3.remove_nodes_from(res2)
        res1 = ICM(G3, patients, 6)
        print(res1)
        sick.append((res1[0]-result[0])/result[0])
        dead.append((res1[1] - result[1])/result[1])
        G3 = build_graph(filename=filenameBC)
    print('------- estimated sick is: ----------')
    print(sum(sick)/len(sick))
    print('------- estimated dead is: ----------')
    print(sum(dead)/len(dead))












