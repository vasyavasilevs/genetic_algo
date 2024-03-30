import random 
import matplotlib.pyplot as plt

#######################################################################

ONE_MAX_LENGHT         = 100 # bit vector for one individ
POPULATION_SIZE        = 200
P_CROSSOVER            = 0.9 # probability of crossover
P_MUTATION             = 0.1
MAX_NUM_OF_GENERATIONS = 50 

RANDOM_SEED            = 42 # for reproducibility of a start random set
random.seed(RANDOM_SEED)

#######################################################################

class FitnessMax():
    def __init__(self):
        self.values = 0

class Individual(list):
        def __init__(self, *args):
            super().__init__(*args) # related to ordinary list
            self.fitness = FitnessMax()
            
def oneMaxFitness(individual) -> int:
    return sum(individual)

def individualCreator():
    return Individual([random.randint(0,1) for i in range(ONE_MAX_LENGHT)])

def populationCreator():
    return [individualCreator() for i in range(POPULATION_SIZE)]
    
def clone(value):
    ind = Individual(value[:])
    ind.fitness.values = value.fitness.values
    return ind

def selTournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0 #indices of 3 individual lists in population
        while i1 == i2 or i1 == i3 or i2 == i3:
            i2, i2, i3 = random.randint(0, p_len - 1),  random.randint(0, p_len - 1), random.randint(0, p_len - 1)
        
        selected_individuals = [population[i1], population[i2], population[i3]]
        
        max_fitness_individual = max(selected_individuals, key=lambda ind: ind.fitness.values)

        offspring.append(max_fitness_individual)

    return offspring

def cxOnePoint(child1, child2):
    s = random.randint(2, len(child1) - 3)
    child1[s:], child2[s:] = child2[s:], child1[s:]
    
def mutFlipBit(mutant, indpb=0.01):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            mutant[indx] = 0 if mutant[indx] == 1 else 1
            
#######################################################################
            
population = populationCreator()
generationCounter = 0

fitnessValues = list(map(oneMaxFitness, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue
            
maxFitnessValues = [] # of indiv in cur pop
meanFitnessValues = [] # of all indiv in current population
            
while max(fitnessValues) < ONE_MAX_LENGHT and generationCounter < MAX_NUM_OF_GENERATIONS:
    generationCounter += 1
    offspring = selTournament(population, len(population))
    offspring = list(map(clone, offspring))
    
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            cxOnePoint(child1, child2)
            
    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutFlipBit(mutant, indpb=1.0/ONE_MAX_LENGHT)
            
    freshFitnessValues = list(map(oneMaxFitness, offspring))
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue
        
    population[:] = offspring
    
    fitnessValues = [ind.fitness.values for ind in population]
    
    maxFitness = max(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)
    print(f"Generation {generationCounter}: Max fitness = {maxFitness}, Mean fitness = {meanFitness}")
    
    best_index = fitnessValues.index(max(fitnessValues))
    print("Best individual = ", population[best_index], "\n")
    
#######################################################################
    
plt.plot(maxFitnessValues, color='orange')
plt.plot(meanFitnessValues, color='olive')
plt.xlabel('Generation')
plt.ylabel('Max/mean fitness')
plt.show()

    