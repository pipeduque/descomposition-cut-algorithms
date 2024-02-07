import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
import numpy as np

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


def decomposition(ns, cs, cs_value):
    memory = {}
    print("Probabilities")
    for probabilitie in probabilities:
        print(probabilitie)
    print("**********************************************************")

    print("Sistema original")
    print(f"{ns}ᵗ⁺¹ | {cs}ᵗ")
    original_system = probabilityTransitionTable(
        cs_to_array(cs), ns_to_array(ns), probabilities
    )
    graphProbability(original_system, "orange", f"{ns}ᵗ⁺¹ | {cs}ᵗ = {cs_value}")
    print("**********************************************************")

    memory = {}

    impresos = set()

    def descomponer(ns, cs, memory):
        if memory.get(cs) is not None and memory.get(cs).get(ns) is not None:
            if any(memory.get(cs).get(ns)):
                print("get in memory")
                return memory.get(cs).get(ns)

        if len(ns) == 1:
            value = probabilityTransitionTable(
                cs_to_array(cs), ns_to_array(ns), probabilities
            )
            return value

        value = []
        for i in range(0, len(ns)):
            if len(value) > 0:
                cross_product = np.kron(value, descomponer(ns[i], cs, memory))
                value = reorder_cross_product(cross_product)

            else:
                value = np.array(descomponer(ns[i], cs, memory))

                if memory.get(cs) == None:
                    memory[cs] = {}

                memory[cs][ns[i]] = value

        return value

    for lenNs in range(len(ns) + 1):
        for i in range(len(ns) - lenNs + 1):
            j = i + lenNs - 1
            ns1, ns2 = ns[i : j + 1], ns[:i] + ns[j + 1 :]

            for lenCs in range(len(cs) + 1):
                for x in range(len(cs) - lenCs + 1):
                    z = x + lenCs - 1
                    cs1, cs2 = cs[x : z + 1], cs[:x] + cs[z + 1 :]

                    # Verificar duplicados
                    combinacion_actual = ((ns1, cs1), (ns2, cs2))
                    combinacion_inversa = ((ns2, cs2), (ns1, cs1))

                    if (
                        combinacion_actual not in impresos
                        and combinacion_inversa not in impresos
                    ) or (ns1 == ns and ns2 == "" and cs1 == "" and cs2 == ""):
                        print(f"({ns2} | {cs2})", f" * ({ns1} | {cs1})")
                        arr1 = np.array(descomponer(ns2, cs2, memory))
                        arr2 = np.array(descomponer(ns1, cs1, memory))

                        partitioned_system = []

                        if len(arr1) > 0 and len(arr2) > 0:
                            cross_product = np.kron(arr1, arr2)
                            partitioned_system = reorder_cross_product(cross_product)

                        elif len(arr1) > 0:
                            partitioned_system = arr1
                        elif len(arr2) > 0:
                            partitioned_system = arr2

                        # Calcular la Distancia de Wasserstein (EMD)
                        emd_distance = wasserstein_distance(
                            original_system, partitioned_system
                        )
                        # print(f"Earth Mover's Distance: {emd_distance}")
                        # graphProbability(partitioned_system, 'blue', f"p({ns2}ᵗ⁺¹|{cs2}ᵗ) * p({ns1}ᵗ⁺¹|{cs1}ᵗ)")

                        impresos.add(combinacion_actual)
                        impresos.add(combinacion_inversa)

    print(memory)


# Ejemplo 4 proyecto
ns = "AC"
cs = "ABC"
cs_value = [1, 0, 0]

decomposition(ns, cs, cs_value)
