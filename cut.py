import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import numpy as np
import networkx as nx

states = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]

probabilities = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
]


def reorder_cross_product(cross_product):
    len_cross = len(cross_product)

    if len_cross == 4:
        new_order = [0, 2, 1, 3]
        return cross_product[new_order]

    elif len_cross == 8:
        new_order = [0, 1, 2, 3, 4, 6, 5, 7]
        return cross_product[new_order]

    return cross_product


def ns_to_array(letras):
    ns_arr = [0] * len(letras)

    return ns_arr


def cs_to_array(cs):
    # Crear un nuevo arreglo con None en las posiciones especificadas por cs
    cs_arr = [cs_value[i] if chr(65 + i) in cs else None for i in range(len(cs_value))]

    return cs_arr


def build_probabilities(probabilities, len_cs):
    extended_probabilities = [None] * len(probabilities)
    for i in range(len(probabilities)):
        extended_probabilities[i] = probabilities[i] + [0] * (
            len_cs - len(probabilities[i])
        )

    return extended_probabilities


def graphProbability(array, color, label):
    labels = [bin(i)[2:] for i in range(len(array))]
    positions = np.arange(len(labels))
    plt.bar(positions, array, color=color, alpha=0.7, label=label)
    for i, valor in enumerate(array):
        plt.text(i, valor + 0.05, f"{valor:.2f}", ha="center", va="bottom")
    plt.xlabel("Next state")
    plt.ylabel("Probability")
    plt.xticks(positions, labels)
    plt.legend()
    plt.ylim(0, 1)
    plt.show()


def getIndicesToMargenalice(states, state):
    availableIndices = []
    indices = {}
    csValue = ""

    for i in range(len(state)):
        if state[i] != None:
            availableIndices.append(i)
            csValue = str(state[i]) + csValue

    for i in range(len(states)):
        key = ""
        for j in range(len(availableIndices)):
            key += str(states[i][availableIndices[j]])

        indices[key] = indices.get(key) + [i] if indices.get(key) else [i]

    if csValue == "":
        return indices, 0

    return indices, int(csValue, 2)


def margenaliceNextState(nsIndices, probabilites):
    nsTransitionTable = [[None] * len(nsIndices) for i in range(len(probabilites))]
    currentColumn = 0
    for indices in nsIndices.values():
        for i in range(len(nsTransitionTable)):
            probability = 0
            for j in range(len(indices)):
                probability += probabilites[i][indices[j]]

            nsTransitionTable[i][currentColumn] = probability

        currentColumn += 1

    return nsTransitionTable


def margenaliceCurrentState(csIndices, nsTransitionTable):
    csTransitionTable = [
        [None] * len(nsTransitionTable[0]) for i in range(len(csIndices))
    ]

    currentRow = 0
    for indices in csIndices.values():
        for i in range(len(csTransitionTable[0])):
            probability = 0
            for j in range(len(indices)):
                probability += nsTransitionTable[indices[j]][i]

            csTransitionTable[currentRow][i] = probability / len(indices)

        currentRow += 1

    return csTransitionTable


def probabilityTransitionTable(currentState, nextState, probabilities):
    result = []
    csTransitionTable = []
    csIndices, csValueIndex = getIndicesToMargenalice(states, currentState)
    missingCs = any(state is None for state in currentState)

    if missingCs:
        for i, state in enumerate(nextState):
            if state is not None:
                newNs = [None] * len(nextState)
                newNs[i] = nextState[i]

                nsIndices, _ = getIndicesToMargenalice(states, newNs)
                nsTransitionTable = margenaliceNextState(nsIndices, probabilities)
                csTransitionTable = margenaliceCurrentState(
                    csIndices, nsTransitionTable
                )
                csValue = csTransitionTable[csValueIndex]

                if len(result) > 0:
                    result = np.kron(result, csValue)
                else:
                    result = csValue

        # result = reorder_cross_product(result)

    else:
        nsIndices, _ = getIndicesToMargenalice(states, nextState)
        nsTransitionTable = margenaliceNextState(nsIndices, probabilities)

        csTransitionTable = margenaliceCurrentState(csIndices, nsTransitionTable)
        result = csTransitionTable[csValueIndex]

    return result


def dfs(G, node, visited, end_node):
    visited.add(node)
    for neighbor in G.neighbors(node):
        if neighbor not in visited:
            dfs(G, neighbor, visited, end_node)

        if end_node in visited:
            return


def is_bipartite(G, start_node, end_node):
    visited = set()
    dfs(G, start_node, visited, end_node)

    return end_node not in visited


def cut_process(ns, cs, cs_value, probabilities, states):
    G = nx.DiGraph()
    add_connections(G, ns, cs)
    print("Grafo principal")
    draw_graph(G)

    nodes = list(G.nodes())
    start_node = nodes[0]
    if is_bipartite(G, start_node, start_node):
        print("El grafo ya está particionado, el proceso termina")
        return

    min_partition = {
        "partitioned_system": [],
        "partition": "",
        "edge_to_remove_1": "",
        "edge_to_remove_2": "",
        "emd": 0,
    }

    start_process(G, ns, cs, cs_value, min_partition, probabilities, states)

    print("**********************************************************")

    edge_to_remove_1 = min_partition["edge_to_remove_1"]
    edge_to_remove_2 = min_partition["edge_to_remove_2"]

    if min_partition["emd"] > 0 and edge_to_remove_1 != "" and edge_to_remove_2 != "":
        G.remove_edge(edge_to_remove_1, edge_to_remove_2)
        G.remove_edge(edge_to_remove_2, edge_to_remove_1)

    print("Grafo final")
    draw_graph(G)

    if min_partition["partition"]:
        print("Minima partición")
        print("emd", min_partition["emd"])
        print("partition", min_partition["partition"])
        graphProbability(
            min_partition["partitioned_system"], "blue", min_partition["partition"]
        )


def start_process(G, ns, cs, cs_value, min_partition, probabilities, states):
    memory = {}
    probabilities = build_probabilities(probabilities, len(states))
    original_system = probabilityTransitionTable(
        cs_to_array(cs), ns_to_array(ns), probabilities
    )

    for i in range(len(ns)):
        nsN = ns[i] + "ᵗ⁺¹"
        for j in range(len(cs)):

            if ns[i] == cs[j]:
                continue

            csC = cs[j] + "ᵗ"

            print("Variable actual", nsN)
            print("cortando", csC, "de", nsN)

            cs_left_cut = cs[:j]
            cs_right_cut = cs[j + 1 :]
            cs_right_partition = cs_left_cut + cs_right_cut

            partition = f"(∅ᵗ⁺¹ | {csC}ᵗ) y ({ns}ᵗ⁺¹ | {cs_right_partition}ᵗ)"
            print("partition: ", partition)

            G.remove_edge(csC, nsN)
            G.remove_edge(nsN, csC)
            draw_graph(G)

            arr1 = np.array(cut("", csC, memory, probabilities))
            arr2 = np.array(cut(ns, cs_right_partition, memory, probabilities))

            partitioned_system = []

            if len(arr1) > 0 and len(arr2) > 0:
                cross_product = np.kron(arr1, arr2)
                partitioned_system = reorder_cross_product(cross_product)
            elif len(arr1) > 0:
                partitioned_system = arr1
            elif len(arr2) > 0:
                partitioned_system = arr2

            # Calcular la Distancia de Wasserstein (EMD)
            emd_distance = wasserstein_distance(original_system, partitioned_system)
            print(f"Earth Mover's Distance: {emd_distance}")

            start_node = csC
            end_node = nsN
            if is_bipartite(G, start_node, end_node):
                print("Bipartición generada")
                if min_partition.get("partition") == "":
                    set_min_partition(
                        min_partition,
                        partition,
                        partitioned_system,
                        emd_distance,
                        csC,
                        nsN,
                    )

                if emd_distance == 0:
                    set_min_partition(
                        min_partition,
                        partition,
                        partitioned_system,
                        emd_distance,
                        csC,
                        nsN,
                    )
                    print("minima partición alcanzada")
                    return

                elif emd_distance <= min_partition.get("emd"):
                    set_min_partition(
                        min_partition,
                        partition,
                        partitioned_system,
                        emd_distance,
                        csC,
                        nsN,
                    )
                    print("minima partición actualizada")

                G.add_edge(csC, nsN)
                G.add_edge(nsN, csC)
                print("bipartición restarurada con costo:", emd_distance)
            else:
                print("No bipartición generada")
                if emd_distance > 0:
                    G.add_edge(csC, nsN)
                    G.add_edge(nsN, csC)
                    print("conexión restarurada con costo: ", emd_distance)
                else:
                    print("conexión elimina sin perdida de información")

            print("----------   ********** ------------")


def set_min_partition(
    min_partition,
    partition,
    partitioned_system,
    emd_distance,
    edge_to_remove_1,
    edge_to_remove_2,
):
    min_partition["partition"] = partition
    min_partition["partitioned_system"] = partitioned_system
    min_partition["emd"] = emd_distance
    min_partition["edge_to_remove_1"] = edge_to_remove_1
    min_partition["edge_to_remove_2"] = edge_to_remove_2


def cut(ns, cs, memory, probabilities):
    if memory.get(cs) is not None and memory.get(cs).get(ns) is not None:
        if any(memory.get(cs).get(ns)):
            return memory.get(cs).get(ns)

    if len(ns) == 1:
        value = probabilityTransitionTable(
            cs_to_array(cs), ns_to_array(ns), probabilities
        )
        return value

    value = []
    for i in range(0, len(ns)):
        if len(value) > 0:
            cross_product = np.kron(value, cut(ns[i], cs, memory, probabilities))
            value = reorder_cross_product(cross_product)

        else:
            value = np.array(cut(ns[i], cs, memory, probabilities))

            if memory.get(cs) == None:
                memory[cs] = {}

            memory[cs][ns[i]] = value

    return value


def add_connections(G, ns, cs):
    for i in range(len(ns)):
        n = ns[i] + "ᵗ⁺¹"

        for j in range(len(cs)):
            if ns[i] == cs[j]:
                continue

            c = cs[j] + "ᵗ"
            G.add_node(n)
            G.add_node(c)
            G.add_edge(c, n)
            G.add_edge(n, c)


def draw_graph(G):
    pos = nx.circular_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=700,
        node_color="skyblue",
        font_size=10,
        font_color="black",
        arrowsize=20,
    )
    plt.show()


states = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]

probabilities = [
    [1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0],
]

# Ejemplo 1 cut conexions
ns = "AB"
cs = "ABC"
cs_value = [1, 0, 0]

cut_process(ns, cs, cs_value, probabilities, states)
