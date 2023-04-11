import random
import numpy as np
import nn
from tqdm import tqdm
import matplotlib.pyplot as plt


class genetic:
    
    def __init__(self, env, numberGenerations, mutationRate, populationSize, numberSteps):
        self.env = env
        self.generations = numberGenerations
        self.populationSize = populationSize
        self.mutationRate = mutationRate
        self.numberSteps = numberSteps

        self.inputSize = 3
        self.hiddenSize = 6
        self.numberHiddenLayers = 3
        self.outputSize = 1
        self.population = self.createRandPopulation()

        self.fitnessHistory = []
        
        

    
    def createRandWeights(self):
        # Randomly initialize weights between -1 and 1
        weights = []
        for i in range(self.numberHiddenLayers - 1):
            if i == 0:
                weights.append(np.random.uniform(-1, 1, size=(self.inputSize, self.hiddenSize)))
            elif i == self.numberHiddenLayers - 1:
                weights.append(np.random.uniform(-1, 1, size=(self.hiddenSize, self.outputSize)))
            else:
                weights.append(np.random.uniform(-1, 1, size=(self.hiddenSize, self.hiddenSize)))

        return weights
    
    def createRandPopulation(self):
        created = []
        for i in range(self.populationSize):
            weights = self.createRandWeights()
            created.append(individual(weights, self.mutationRate))
        return created
    
    def crossover(self, bestPerformers): # arithmetic crossover
        newPopulation = []
        for i in range(self.populationSize - len(bestPerformers)):
            parent1 = bestPerformers[random.randint(0, len(bestPerformers) - 1)]
            parent2 = bestPerformers[random.randint(0, len(bestPerformers) - 1)]
            weights = []

            for i in range(len(parent1.weights)):
                weights.append((parent1.weights[i] + parent2.weights[i]) / 2)
            child = individual(weights, self.mutationRate)
            newPopulation.append(child)
        self.population = newPopulation
    
    def addBestPerformers(self, bestPerformers):
        for individual in bestPerformers:
            self.population.append(individual)


    def mutatePopulation(self):
        for individual in self.population:
            individual.mutate()

    def rejectOutliers(self, data, m=2):
        return data[abs(data - np.mean(data)) < m * np.std(data)]
    
    def saveModel(self):
        for i, weight in enumerate(self.population[0].weights):
            np.save(f'w{i}.npy', self.population[0].weights[0])
        np.savetxt('data.csv', self.fitnessHistory, delimiter=',')
        
        #data = self.rejectOutliers(np.asarray(self.fitnessHistory))
        self.fitnessHistory.pop(0)
        plt.plot(self.fitnessHistory)
        plt.savefig('fitnessHistory.png')
    
    def loadModel(self, layers):
        weights = []
        for i in range(layers):
            weights.append(np.load(f'w{i}.npy'))
            print(f'Loaded {i+1} layers')
        
        if len(self.population) == 0:
            self.population = []
        for i in range(self.populationSize):
            self.population.append(individual(weights, self.mutationRate))
    
    def run(self):
        self.fitnessHistory = []
        for gen in tqdm(range(self.generations)):
            for individual in self.population:
                initialObs, info = self.env.reset(seed=100) # Seed remains constant so that the same initial observation is used for each individual
                net = nn.nn(individual.weights)
                observation = initialObs
                for i in (range(self.numberSteps)):
                    nextStep = net(observation)[0]
                    observation, fitness, term, trun, info = self.env.step(nextStep)
                    individual.setFitness(fitness)
                    # if observation[0] < 0:
                    #     break                
            bestPerformers = sorted(self.population, key=lambda x: x.getAverageFitness(), reverse=True)[:int(10)]
            self.fitnessHistory.append(bestPerformers[0].getAverageFitness())
            self.crossover(bestPerformers)
            self.mutatePopulation()
            #self.addBestPerformers(bestPerformers)

    def predict(self, numberSteps, seed):
        initialObs, info = self.env.reset(seed=seed)
        net = nn.nn(self.population[0].weights)
        observation = initialObs
        for i in (range(numberSteps)):
            nextStep = net(observation)[0]
            print(nextStep)
            observation, fitness, term, trun, info = self.env.step(nextStep)
            self.env.step(nextStep)
            self.env.render()

                






class individual:

    def __init__(self, weights, mutationRate):
        self.weights = weights
        self.mutationRate = mutationRate
        self.fitnessHistory = []

    def setFitness(self, fitness):
        self.fitness = fitness
        self.fitnessHistory.append(fitness)

    def getAverageFitness(self):
        return np.average(np.asarray(self.fitnessHistory))
    
    def mutate(self):
        newWeights = []
        for weight in self.weights:
            mutationFactor = np.random.uniform(-self.mutationRate, self.mutationRate, size=weight.shape)
            newWeight = weight * (1 + mutationFactor)
            newWeights.append(newWeight)
        self.weights = newWeights
                
        
