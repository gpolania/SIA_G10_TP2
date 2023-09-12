from math import *
import math
import random
import json

from matplotlib import pyplot as plt

def strength(items):
    return 100*tanh(0.01*items)

def agility(items):
    return tanh(0.01*items)

def expertise(items):
    return 0.6*tanh(0.01*items)

def endurance(items):
    return tanh(0.01*items)

def liveliness(items):
    return 100*tanh(0.01*items)



def chooseRandomGenes(p, l):
    genes = ["agility", "expertise", "strength", "endurance","liveliness", "height"]

    for _ in range(l):
        result.append(genes[p])
        p = (p + 1) % len(genes)

class Character:

    def __init__(self, chromossome):
        self.chromossome = chromossome
        self.height = self.chromossome["height"]
        self.items = {key: value for key, value in self.chromossome.items() if key != "height"}
        self.ATM = 0.5 - (3*self.chromossome["height"] - 5)**4 + (3*self.chromossome["height"] - 5)**2 + self.chromossome["height"]/2
        self.DEM = 2 + (3*self.chromossome["height"] - 5)**4 - (3*self.chromossome["height"] - 5)**2 - self.chromossome["height"]/2
        self.attack = (agility(self.chromossome["agility"]) + expertise(self.chromossome["expertise"])) * strength(self.chromossome["strength"]) * self.ATM
        self.defense = (endurance(self.chromossome["endurance"]) + expertise(self.chromossome["expertise"])) * liveliness(self.chromossome["liveliness"]) * self.DEM


    def changeGenome(self, gene, newValue):
        self.chromossome[gene] = newValue
        if newValue == "height":
            self.height = newValue
        else:
            self.items[gene] = newValue

        self.ATM = 0.5 - (3*self.chromossome["height"] - 5)**4 + (3*self.chromossome["height"] - 5)**2 + self.chromossome["height"]/2
        self.DEM = 2 + (3*self.chromossome["height"] - 5)**4 - (3*self.chromossome["height"] - 5)**2 - self.chromossome["height"]/2
        self.attack = (agility(self.chromossome["agility"]) + expertise(self.chromossome["expertise"])) * strength(self.chromossome["strength"]) * self.ATM
        self.defense = (endurance(self.chromossome["endurance"]) + expertise(self.chromossome["expertise"])) * liveliness(self.chromossome["liveliness"]) * self.DEM

    def equalGeneme(self, other, threshold):
        return abs(self.height - other.height) < threshold and abs(self.items["agility"] - other.items["agility"]) < threshold and abs(self.items["expertise"] - other.items["expertise"]) < threshold and abs(self.items["strength"] - other.items["strength"]) < threshold and abs(self.items["endurance"] - other.items["endurance"]) < threshold and abs(self.items["liveliness"] - other.items["liveliness"]) < threshold

    def __repr__(self):
        return "height: " + str(self.height) + ", attack: " + str(self.attack) + ", defense: " + str(self.defense) + "\nitems: " + str(self.items) + '\n\n'


class Warrior(Character):

    def __init__(self, chromossome):
        super().__init__(chromossome)
        self.performance = 0.6 * self.attack + 0.4 * self.defense


class Archer(Character):

    def __init__(self, chromossome):
        super().__init__(chromossome)
        self.performance = 0.9 * self.attack + 0.1 * self.defense

class Defender(Character):

    def __init__(self, chromossome):
        super().__init__(chromossome)
        self.performance = 0.1 * self.attack + 0.9 * self.defense


class Infiltrate(Character):

    def __init__(self, chromossome):
        super().__init__(chromossome)
        self.performance = 0.7 * self.attack + 0.3 * self.defense


class Population:

    def __init__(self, size, character, generation, individuals):
        self.size = size
        self.individuals = individuals
        self.generation = generation
        self.character = character

    def getIndividual(self, index):
        return self.individuals[index]

class GeneticAlgorithm:

    characterMap = {
        "Warrior": Warrior,
        "Archer": Archer,
        "Defender": Defender,
        "Infiltrate": Infiltrate,
    }


    def __init__(self, config_file):

        # Load configuration from the JSON file
        with open(config_file, 'r') as config:
            self.config = json.load(config)

        self.character = self.characterMap[self.config["character_class"]]


    def initializePopulation(self):
        individuals = []
        populationSize = self.config["genetic_algorithm"]["population_size"]
        character = self.characterMap[ self.config["character_class"]]
        for _ in range(populationSize):
            chromossome = {}
            chromossome["agility"] = random.uniform(0, 150)
            chromossome["expertise"] = random.uniform(0, 150 - chromossome["agility"])
            chromossome["strength"] = random.uniform(0, 150 - chromossome["agility"] - chromossome["expertise"])
            chromossome["endurance"] = random.uniform(0, 150 - chromossome["agility"] - chromossome["expertise"] - chromossome["strength"])
            chromossome["liveliness"] = 150 - chromossome["agility"] - chromossome["expertise"] - chromossome["strength"] - chromossome["endurance"]
            chromossome["height"] = random.uniform(1.3, 2)
            individuals += [character(chromossome)]

        self.currentPopulation = Population(populationSize, character, 0, individuals)
        self.nAlleles = len(self.currentPopulation.getIndividual(0).chromossome)

    # Selections
    def eliteSelection(self):
        N = self.currentPopulation.size
        k = self.config["genetic_algorithm"]["k"]
        # Ordenar la población por aptitud de mayor a menor
        sorted_population = sorted(self.currentPopulation.individuals, key=lambda x: x.performance, reverse=True)

        # Inicializar la lista de individuos seleccionados
        selected_individuals = []

        # Calcular n(i) y seleccionar individuos
        for i, individual in enumerate(sorted_population):
            n_i = math.ceil((k - i) / N)
            selected_individuals.extend([individual] * int(n_i))

        # Asegurarse de que la cantidad seleccionada sea igual a K
        selected_individuals = selected_individuals[:k]

        return selected_individuals

    def rouletteWheelSelection(self):
        # Calcula las aptitudes relativas (pi) y acumuladas (qi)
        population = self.currentPopulation.individuals
        k = self.config["genetic_algorithm"]["k"]

        total_performance = sum(individual.performance for individual in population)
        relative_performance = [individual.performance / total_performance for individual in population]
        cumulative_performance = [sum(relative_performance[:i + 1]) for i in range(len(relative_performance))]

        # Selecciona k individuos utilizando ruleta acumulada
        selected_individuals = []
        for _ in range(k):
            r = random.uniform(0, 1)
            for i, qi in enumerate(cumulative_performance):
                if i > 0:
                    qi_minus_1 = cumulative_performance[i - 1]
                else:
                    qi_minus_1 = 0

                if qi_minus_1 < r <= qi:
                    selected_individual = population[i]
                    break
            if selected_individual is not None:
                selected_individuals.append(selected_individual)
        return selected_individuals

    def universalSelection(self):

        population = self.currentPopulation.individuals
        k = self.config["genetic_algorithm"]["k"]

        # Calcula las aptitudes relativas (pi) y acumuladas (qi)
        total_performance = sum(individual.performance for individual in population)
        relative_performance = [individual.performance / total_performance for individual in population]
        cumulative_performance = [sum(relative_performance[:i + 1]) for i in range(len(relative_performance))]

        # Selecciona k individuos utilizando ruleta acumulada
        selected_individuals = []
        r = random.uniform(0, 1)
        for j in range(k):

            rj=(r+j)/k

            for i, qi in enumerate(cumulative_performance):
                if i > 0:
                    qi_minus_1 = cumulative_performance[i - 1]
                else:
                    qi_minus_1 = 0

                if qi_minus_1 < rj <= qi:
                    selected_individual = population[i]
                    break
            if selected_individual is not None:
                selected_individuals.append(selected_individual)
        return selected_individuals

    def boltzmannSelection(self):

        k = self.config["genetic_algorithm"]["k"]

        population = self.currentPopulation.individuals
        To = 100
        Tc = 0.01
        c = 0.05
        t=self.currentPopulation.generation
        temperature = Tc + (To - Tc) * math.exp(-c * t)

        total_exp=sum(math.exp(individual.performance)/temperature for individual in population)
        for individual in population:
            individual.performance=math.exp(individual.performance / temperature) / total_exp 
        

        total_performance = sum(individual.performance for individual in population)
        relative_performance = [individual.performance / total_performance for individual in population]
        cumulative_performance = [sum(relative_performance[:i + 1]) for i in range(len(relative_performance))]

        # Se realiza ruleta
        selected_individuals = []
        for _ in range(k):
            r = random.uniform(0, 1)
            for i, qi in enumerate(cumulative_performance):
                if i > 0:
                    qi_minus_1 = cumulative_performance[i - 1]
                else:
                    qi_minus_1 = 0

                if qi_minus_1 < r <= qi:
                    selected_individual = population[i]
                    break
            if selected_individual is not None:
                selected_individuals.append(selected_individual)
        return selected_individuals
    def tournamentDeterministicSelection(self):

        population = self.currentPopulation.individuals
        k = self.config["genetic_algorithm"]["k"]
        m=10 #configurar numero de invididuos escogidos al azar
        selected_individuals = []

        while len(selected_individuals) < k:
            # Seleccionar M individuos al azar
            tournament = random.sample(population, m)

            # Elegir el mejor individuo del torneo
            best_individual = max(tournament, key=lambda x: x.performance)

            selected_individuals.append(best_individual)

        return selected_individuals
    def tournamentProbabilisticSelection(self):

        population = self.currentPopulation.individuals
        k = self.config["genetic_algorithm"]["k"]

        selected_individuals = []

        threshold=0.7 #Se elige un valor de Threshold en [0.5 , 1]

        while len(selected_individuals) < k:

            tournament = random.sample(population, 2)


            r = random.uniform(0, 1)

            # Comparar r con el Threshold
            if r < threshold:
                # Seleccionar el individuo más apto
                selected_individual = max(tournament, key=lambda x: x.performance)
            else:
                # Seleccionar el individuo menos apto
                selected_individual = min(tournament, key=lambda x: x.performance)

            selected_individuals.append(selected_individual)
        return selected_individuals
    def rankBasedSelection(self):

        population = self.currentPopulation.individuals
        k = self.config["genetic_algorithm"]["k"]
        N = self.currentPopulation.size

        # Calcular el ranking de aptitud real de cada individuo
        rankings = list(range(1, N+1))

        # Calcular las pseudo-aptitudes utilizando la fórmula dada
        pseudo_aptitudes = [(N - rank) / N for rank in rankings]

        # Ordenar la población actual por rendimiento
        sorted_population = sorted(population, key=lambda x: x.performance, reverse=True)

        for i in range(len(sorted_population)):
            sorted_population[i].performance = pseudo_aptitudes[i]

        total_performance = sum(individual.performance for individual in sorted_population)
        relative_performance = [individual.performance / total_performance for individual in sorted_population]
        cumulative_performance = [sum(relative_performance[:i + 1]) for i in range(len(relative_performance))]

        # Se realiza ruleta
        selected_individuals = []
        for _ in range(k):
            r = random.uniform(0, 1)
            for i, qi in enumerate(cumulative_performance):
                if i > 0:
                    qi_minus_1 = cumulative_performance[i - 1]
                else:
                    qi_minus_1 = 0

                if qi_minus_1 < r <= qi:
                    selected_individual = population[i]
                    break
            if selected_individual is not None:
                selected_individuals.append(selected_individual)
        return selected_individuals
    # Crossovers
    def singlePointCrossover(self, parent1, parent2):

        gene = random.choices(["agility", "expertise", "strength", "endurance","liveliness", "height"], k=1)[0]

        offspring1Chromossome = parent1.chromossome
        offspring2Chromossome = parent2.chromossome


        aux = offspring1Chromossome[gene]
        offspring1Chromossome[gene] = offspring2Chromossome[gene]
        offspring2Chromossome[gene] = aux

        return self.character(offspring1Chromossome), self.character(offspring2Chromossome)


    def twoPointCrossover(self, parent1, parent2):
        twoGenes = random.choices(["agility", "expertise", "strength", "endurance","liveliness", "height"], k=2)

        # Ensure the two numbers are different
        while twoGenes[0] == twoGenes[1]:
            twoGenes = random.choices(["agility", "expertise", "strength", "endurance","liveliness", "height"], k=2)


        offspring1Chromossome = parent1.chromossome
        offspring2Chromossome = parent2.chromossome

        aux = offspring1Chromossome[twoGenes[0]], offspring2Chromossome[twoGenes[0]]
        offspring1Chromossome[twoGenes[0]], offspring1Chromossome[twoGenes[1]] = offspring2Chromossome[twoGenes[0]], offspring2Chromossome[twoGenes[1]]
        offspring2Chromossome[twoGenes[0]], offspring2Chromossome[twoGenes[1]] = aux
        return self.character(offspring1Chromossome), self.character(offspring2Chromossome)


    def ringCrossover(self, parent1, parent2):

        p = random.randint(0, self.nAlleles-1)
        l = random.randint(0, self.nAlleles-1)

        genes = ["agility", "expertise", "strength", "endurance","liveliness", "height"]
        res = []
        for _ in range(l): # circular
            res.append(genes[p])
            p = (p + 1) % len(genes)

        offspring1Chromossome = parent1.chromossome
        offspring2Chromossome = parent2.chromossome

        for gene in res:

            offspring1Chromossome[gene] = parent2.chromossome[gene]
            offspring2Chromossome[gene] = parent1.chromossome[gene]

        return self.character(offspring1Chromossome), self.character(offspring2Chromossome)

    def uniformCrossover(self, parent1, parent2):
        prob = 0.5

        offspring1Chromossome = parent1.chromossome
        offspring2Chromossome = parent2.chromossome

        for gene in ["agility", "expertise", "strength", "endurance","liveliness", "height"]:

            if random.choices([0, 1], weights=[1-prob, prob], k=1)[0]: # will switch whith probability = prob
                offspring1Chromossome[gene] = parent2.chromossome[gene]
                offspring2Chromossome[gene] = parent1.chromossome[gene]

        return self.character(offspring1Chromossome), self.character(offspring2Chromossome)



    def mutate(self, character, Pm):

        if "gen" == self.config["genetic_algorithm"]["mutation_method"]:
            if random.random() < Pm:
                gene = random.choices(["agility", "expertise", "strength", "endurance","liveliness", "height"], k=1)[0]
                character.changeGenome(gene, random.uniform(0, 150))

        elif "multigen" == self.config["genetic_algorithm"]["mutation_method"]:
            for gene in ["agility", "expertise", "strength", "endurance","liveliness", "height"]:
                if random.random() < Pm:
                    character.changeGenome(gene, random.uniform(0, 150))


    selectionMap = {
        "elite": eliteSelection,
        "roulette": rouletteWheelSelection,
        "universal": universalSelection,
        "boltzmann": boltzmannSelection,
        "tournament_deterministic": tournamentDeterministicSelection,
        "tournament_probabilistic": tournamentProbabilisticSelection,
        "rank_based": rankBasedSelection,
    }
    crossoverMap = {
        "single_point": singlePointCrossover,
        "two_point": twoPointCrossover,
        "ring": ringCrossover,
        "uniform": uniformCrossover,
    }
    def run(self):
        performance_history = []
        self.initializePopulation()
        selectionM1 = self.selectionMap[self.config["genetic_algorithm"]["selection_methods"]["method1"]]
        selectionM2 = self.selectionMap[self.config["genetic_algorithm"]["selection_methods"]["method2"]]
        selectionW1 = self.config["genetic_algorithm"]["selection_weights"]["method1"]
        replacementM1 = self.selectionMap[self.config["genetic_algorithm"]["replacement_methods"]["method3"]]
        replacementM2 = self.selectionMap[self.config["genetic_algorithm"]["replacement_methods"]["method4"]]
        replacementW1 = self.config["genetic_algorithm"]["replacement_weights"]["method3"]
        crossover = self.crossoverMap[self.config["genetic_algorithm"]["crossover_method"]]

        terminationCriteria = self.config["genetic_algorithm"]["termination_criteria"]
        if terminationCriteria == "max_generations":
            iter = 0
            while iter < self.config["genetic_algorithm"]["max_generations"]:
                numSelectedM1 = int(selectionW1 * self.currentPopulation.size)
                numSelectedM2 = self.currentPopulation.size - numSelectedM1

                selectedIndividuals = selectionM1(self)[:numSelectedM1] + selectionM2(self)[:numSelectedM2]

                # Crossover
                offspring = []
                for i in range(0, self.currentPopulation.size - 1, 2):
                    if i + 1 < len(selectedIndividuals):
                        parent1 = selectedIndividuals[i]
                        parent2 = selectedIndividuals[i + 1]

                        offspring1, offspring2 = crossover(self, parent1, parent2)
                        offspring.extend([offspring1, offspring2])

                # Mutation
                for individual in offspring:
                    self.mutate(individual, self.config["genetic_algorithm"]["mutation_rate"])


                # Replacement
                numReplacedM1 = int(replacementW1*self.currentPopulation.size)
                numReplacedM2 = self.currentPopulation.size - numReplacedM1

                replacedIndividuals = replacementM1(self)[:numReplacedM1] + replacementM2(self)[:numReplacedM2]

                self.currentPopulation = Population(self.currentPopulation.size, self.character, self.currentPopulation.generation + 1, replacedIndividuals)

                print(f"Generación {iter + 1}:")
                for i, individual in enumerate(self.currentPopulation.individuals):
                    print(f"Individuo {i + 1}:")
                    print(f"  Desempeño: {individual.performance}")
                    print(f"  Información del individuo: {individual}")

                iter += 1

                generation_performance = [individual.performance for individual in self.currentPopulation.individuals]
                average_performance = sum(generation_performance) / len(generation_performance)
                performance_history.append(average_performance)

            # Generar la gráfica después de finalizar todas las generaciones
            plt.plot(performance_history)
            plt.xlabel('Generaciones')
            plt.ylabel('Promedio de Desempeño')
            plt.title('Evolución del Promedio de Desempeño a lo Largo de las Generaciones')
            plt.show()


        elif terminationCriteria == "structure":
            thresholdGene = 0.1

            numEqualGenes = 0
            thresholdEqualInd = int(0.8*self.currentPopulation.size)
            iter = 0
            while numEqualGenes < thresholdEqualInd:
                numSelectedM1 = int(selectionW1 * self.currentPopulation.size)
                numSelectedM2 = self.currentPopulation.size - numSelectedM1

                selectedIndividuals = selectionM1(self)[:numSelectedM1] + selectionM2(self)[:numSelectedM2]

                # Crossover
                offspring = []
                for i in range(0, self.currentPopulation.size - 1, 2):
                    if i + 1 < len(selectedIndividuals):
                        parent1 = selectedIndividuals[i]
                        parent2 = selectedIndividuals[i + 1]

                        offspring1, offspring2 = crossover(self, parent1, parent2)
                        offspring.extend([offspring1, offspring2])

                # Mutation
                for individual in offspring:
                    self.mutate(individual, self.config["genetic_algorithm"]["mutation_rate"])


                # Replacement
                numReplacedM1 = int(replacementW1*self.currentPopulation.size)
                numReplacedM2 = self.currentPopulation.size - numReplacedM1

                replacedIndividuals = replacementM1(self)[:numReplacedM1] + replacementM2(self)[:numReplacedM2]

                self.currentPopulation = Population(self.currentPopulation.size, self.character, self.currentPopulation.generation + 1, replacedIndividuals)

                print(f"Generación {iter + 1}:")
                for i, individual in enumerate(self.currentPopulation.individuals):
                    print(f"Individuo {i + 1}:")
                    print(f"  Desempeño: {individual.performance}")
                    print(f"  Información del individuo: {individual}")

                iter += 1

                for i, individual1 in enumerate(self.currentPopulation.individuals):
                    for individual2 in self.currentPopulation.individuals[i+1:]:
                        if individual1.equalGeneme(individual2,thresholdGene):
                            numEqualGenes += 1

                generation_performance = [individual.performance for individual in self.currentPopulation.individuals]
                average_performance = sum(generation_performance) / len(generation_performance)
                performance_history.append(average_performance)

            # Generar la gráfica después de finalizar todas las generaciones
            plt.plot(performance_history)
            plt.xlabel('Generaciones')
            plt.ylabel('Promedio de Desempeño')
            plt.title('Evolución del Promedio de Desempeño a lo Largo de las Generaciones')
            plt.show()

        elif terminationCriteria == "content":
            thresholdFitness = 0.1
            thresholdEqualFitness = int(0.2 * self.currentPopulation.size)
            iter = 0
            bestFitness = 0  # Inicializa con un valor muy bajo

            while iter < thresholdEqualFitness:
                numSelectedM1 = int(selectionW1 * self.currentPopulation.size)
                numSelectedM2 = self.currentPopulation.size - numSelectedM1

                selectedIndividuals = selectionM1(self)[:numSelectedM1] + selectionM2(self)[:numSelectedM2]

                # Crossover
                offspring = []
                for i in range(0, self.currentPopulation.size - 1, 2):
                    if i + 1 < len(selectedIndividuals):
                        parent1 = selectedIndividuals[i]
                        parent2 = selectedIndividuals[i + 1]

                        offspring1, offspring2 = crossover(self, parent1, parent2)
                        offspring.extend([offspring1, offspring2])

                # Mutation
                for individual in offspring:
                    self.mutate(individual, self.config["genetic_algorithm"]["mutation_rate"])

                # Replacement
                numReplacedM1 = int(replacementW1 * self.currentPopulation.size)
                numReplacedM2 = self.currentPopulation.size - numReplacedM1

                replacedIndividuals = replacementM1(self)[:numReplacedM1] + replacementM2(self)[:numReplacedM2]

                self.currentPopulation = Population(self.currentPopulation.size, self.character, self.currentPopulation.generation + 1, replacedIndividuals)

                print(f"Generación {iter + 1}:")
                for i, individual in enumerate(self.currentPopulation.individuals):
                    print(f"Individuo {i + 1}:")
                    print(f"  Desempeño: {individual.performance}")
                    print(f"  Información del individuo: {individual}")

                generation_performance = [individual.performance for individual in self.currentPopulation.individuals]
                average_performance = sum(generation_performance) / len(generation_performance)
                performance_history.append(average_performance)

                # Actualizar el mejor fitness en cada generación
                currentBestFitness = max(generation_performance)
                if currentBestFitness <= bestFitness:
                    iter += 1
                else:
                    bestFitness = currentBestFitness

            # Generar la gráfica después de finalizar todas las generaciones
            plt.plot(performance_history)
            plt.xlabel('Generaciones')
            plt.ylabel('Promedio de Desempeño')
            plt.title('Evolución del Promedio de Desempeño a lo Largo de las Generaciones')
            plt.show()


            # Generar la gráfica después de finalizar todas las generaciones
            plt.plot(performance_history)
            plt.xlabel('Generaciones')
            plt.ylabel('Promedio de Desempeño')
            plt.title('Evolución del Promedio de Desempeño a lo Largo de las Generaciones')
            plt.show()

if __name__ == "__main__":

    config_file = "1puento_guerreiro.json"  # Update with the path to your configuration file
    genetic_algorithm = GeneticAlgorithm(config_file)
    
    genetic_algorithm.run()
    
