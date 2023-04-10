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
        self.hiddenSize = 2
        self.outputSize = 1
        self.population = self.createRandPopulation()

        
        

    
    def createRandWeights(self):
        # Randomly initialize weights between -1 and 1
        w1 = np.random.uniform(-1, 1, size=(self.inputSize, self.hiddenSize))
        w2 = np.random.uniform(-1, 1, size=(self.hiddenSize, self.outputSize))

        return w1, w2
    
    def createRandPopulation(self):
        created = []
        for i in range(self.populationSize):
            w1, w2 = self.createRandWeights()
            created.append(individual(w1, w2, self.mutationRate))
        return created
    
    def crossover(self, bestPerformers): # arithmetic crossover
        newPopulation = []
        for i in range(self.populationSize):
            parent1 = bestPerformers[random.randint(0, len(bestPerformers) - 1)]
            parent2 = bestPerformers[random.randint(0, len(bestPerformers) - 1)]
            w1 = (parent1.w1 + parent2.w1) / 2
            w2 = (parent1.w2 + parent2.w2) / 2
            child = individual(w1, w2, self.mutationRate)
            newPopulation.append(child)
        self.population = newPopulation

    def mutatePopulation(self):
        for individual in self.population:
            individual.mutate()
    
    def run(self):
        fitnessHistory = []
        for gen in tqdm(range(self.generations)):
            for individual in self.population:
                initialObs, info = self.env.reset(seed=0) # Seed remains constant so that the same initial observation is used for each individual
                net = nn.nn(individual.w1, individual.w2)
                observation = initialObs
                for i in (range(self.numberSteps)):
                    nextStep = net(observation)[0]
                    observation, fitness, term, trun, info = self.env.step(nextStep)
                    individual.setFitness(fitness)
                    # if observation[0][1] > 0:
                    #     break
                

            bestPerformers = sorted(self.population, key=lambda x: x.getAverageFitness(), reverse=True)[:int(self.populationSize / 2)]
            fitnessHistory.append(bestPerformers[0].getAverageFitness())
            self.crossover(bestPerformers)
            self.mutatePopulation()

        plt.plot(fitnessHistory)

                






class individual:

    def __init__(self, w1, w2, mutationRate):
        self.w1 = w1
        self.w2 = w2
        self.mutationRate = mutationRate
        self.fitnessHistory = []

    def setFitness(self, fitness):
        self.fitness = fitness
        self.fitnessHistory.append(fitness)

    def getAverageFitness(self):
        return np.average(np.asarray(self.fitnessHistory))
    
    def mutate(self):
        mutationFactor1 = np.random.uniform(-self.mutationRate, self.mutationRate, size=self.w1.shape)
        mutationFactor2 = np.random.uniform(-self.mutationRate, self.mutationRate, size=self.w2.shape)

        self.w1 = self.w1 * (1 + mutationFactor1)
        self.w2 = self.w2 * (1 + mutationFactor2)
        
