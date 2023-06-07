import random
import numpy as np
import nn
import pickle
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
        self.hiddenSize = 12
        self.numberHiddenLayers = 2
        self.outputSize = 1
        self.population = self.createRandPopulation()

        self.bestPerformers = []
        self.fitnessHistory = []
        
        

    
    def createRandWeights(self):
        # Randomly initialize weights between -1 and 1
        weights = []
        for i in range(self.numberHiddenLayers + 1):
            if i == 0:
                weights.append(np.random.uniform(-1, 1, size=(self.inputSize, self.hiddenSize)))
            elif i == self.numberHiddenLayers:
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
        if len(self.bestPerformers) == 0:
            print("there are no best performers to save")
            return
        
        with open("weights.pkl", 'wb') as f:
            pickle.dump(self.bestPerformers[0].weights, f)
        #np.save('weights.npy', self.bestPerformers[0].weights)
        np.savetxt('data1.csv', self.fitnessHistory, delimiter=',')
        
        #data = self.rejectOutliers(np.asarray(self.fitnessHistory))
        plt.plot(self.fitnessHistory)
        plt.savefig('fitnessHistory.png')
    
    def loadModel(self, path):
        weights = []
        if path is not None:
            with open(f'{path}/weights.pkl', 'rb') as f:
                weights = pickle.load(f)

            for i in range(len(weights)):
                print(np.asarray(weights[i]).shape)

            if len(self.population) == 0:
                for i in range(self.populationSize):
                    self.population.append(individual(weights, self.mutationRate))
            else:
                for individual in self.population:
                    individual.weights = weights

            print('Model loaded successfully')
        else:
            print('No path specified')
    
    def run(self, seed):
        self.fitnessHistory = []
        for gen in tqdm(range(self.generations)):
            for individual in self.population:
                initialObs, info = self.env.reset(seed=seed) # Seed remains constant so that the same initial observation is used for each individual
                net = nn.nn(individual.weights)
                observation = initialObs
                for i in (range(self.numberSteps)):
                    nextStep = net(observation)[0]
                    observation, fitness, term, trun, info = self.env.step(nextStep)
                    individual.setFitness(observation, fitness)
                    # if observation[0] < 0:
                    #     break                
            self.bestPerformers = sorted(self.population, key=lambda x: x.getFitness(), reverse=True)[:int(20)]
            self.fitnessHistory.append(self.bestPerformers[0].getFitness())
            self.crossover(self.bestPerformers)
            self.mutatePopulation()
            #self.addBestPerformers(bestPerformers)

    def predict(self, numberSteps, seed):
        initialObs, info = self.env.reset(seed=seed)
        net = nn.nn(self.population[0].weights)
        observation = initialObs
        for i in (range(numberSteps)):
            nextStep = net(observation)[0]
            observation, fitness, term, trun, info = self.env.step(nextStep)
            self.env.step(nextStep)
            self.env.render()

                






class individual:

    def __init__(self, weights, mutationRate):
        self.weights = weights
        self.mutationRate = mutationRate
        self.fitnessHistory = []

    def setFitness(self, observation, fitness):
        # angle = abs(observation[0])
        # angular_velocity = observation[2]
        # reward = -angle - abs(angular_velocity)*.15
        
        # # Option 2: add a time penalty
        # if angle > 0.15:
        #     reward -= 0.45
        
        self.fitnessHistory.append(fitness)


    def getFitness(self):
        return np.average(np.asarray(self.fitnessHistory))
    
    def mutate(self):
        newWeights = []
        for weight in self.weights:
            mutationFactor = np.random.uniform(-self.mutationRate, self.mutationRate, size=weight.shape)
            newWeight = weight * (1 + mutationFactor)
            newWeights.append(newWeight)
        self.weights = newWeights
                
        
