import numpy as np
import matplotlib.pyplot as plt
import random

#xs
xs = np.arange(0, 1, 0.01)

#bases
def get_base(x0, sig):
    return 1.0 / (np.sqrt(2 * np.pi * sig)) * np.exp(
        -(xs - x0) * (xs - x0) / (2 * sig * sig)
    )

# B1 = get_base(0.2, 0.02)
# B2 = get_base(0.5, 0.02)
# B3 = get_base(0.9, 0.02)
# B4 = get_base(0.65, 0.02)
# B5 = get_base(0.32, 0.02)

#bases to be used
bases = []
for _ in range(10):
    x0 = random.random()
    sig = random.uniform(0.01, 0.1)
    bases.append(get_base(x0, sig))


#desired output -- testing purposes!! You should put the result of the expt here!!
coeffs = [random.random() for _ in range(len(bases))]
Desired = [a * B for a, B in zip(coeffs, bases)]
Desired = sum(Desired)

class Indicator:

    def __init__(self, *bases):
        self.bases = [b for b in bases]
        self.W = np.random.random(len(bases))
        #self.W /= sum(self.W)

    def output(self):
        S = 0.0
        for i in range(len(self.bases)):
            S += self.bases[i] * self.W[i]

        return S

    def calculateCost(self, desired):
        guess = self.output()
        guess -= guess[0]
        SSE = (guess - desired) * (guess  - desired)
        #find a better cost function!!
        self.cost = sum(SSE)/len(SSE)

    def calculateFitness(self):
        self.fitness = 1.0 - self.cost

    def mutate(self, rate=0.1):
        if np.random.random() < rate:
            index = np.random.randint(len(self.W))
            self.W[index] += random.uniform(-0.1, 0.1)
        self.W = np.clip(self.W, 0.0, 1.0)

    def set_W(self, W):
        self.W = W

    def clone(self):
        newI = Indicator(*self.bases)
        newI.set_W(self.W)
        return newI


class PopulationManager:

    def __init__(self, *bases, popsize = 10):
        self.popsize = popsize
        self.create_pop(*bases)

    def create_pop(self, *bases):
        population = []
        for _ in range(self.popsize):
            I = Indicator(*bases)
            population.append(I)

        self.population = population

    def NextGeneration(self, desired):

        for I in self.population:
            I.calculateCost(desired)

        #normalising the cost value!
        total = 0.0
        for I in self.population:
            total += I.cost
        for I in self.population:
            I.cost /= total

        #it is easier to work in terms of fitness!!
        ##not using it now!! Do we need this??
        for I in self.population:
            I.calculateFitness()

        newpop = []
        for _ in range(self.popsize):
            # child = self.makeChild(self.population)
            child = self.get_BestIndicator(desired).clone()
            child.mutate(0.01)
            newpop.append(child)

        return newpop

    def makeChild(self, population): #Do we need this???
        index = 0
        r = np.random.random()
        while r > 0.0:
            r -= population[index].fitness
            index += 1
        index -= 1

        return population[index].clone()

    def evolve(self, desired, generations=1):
        for _ in range(generations):
            self.population = self.NextGeneration(desired)

    def get_BestIndicator(self, desired):

        for I in self.population:
            I.calculateCost(desired)

        best = self.population[0]
        bestcost = self.population[0].cost

        for i in range(1, len(self.population)):
            if self.population[i].cost < bestcost:
                best = self.population[i]
                bestcost = self.population[i].cost

        return best



#main
plt.ion()

plt.plot(xs, Desired, label="desired")

pop = PopulationManager(*bases, 100)
# pop.evolve(Desired, 500)
out = pop.get_BestIndicator(Desired).output()
outplot, = plt.plot(xs, out-out[0], label="prediction")
for i in range(5000):
    pop.population = pop.NextGeneration(Desired)
    best = pop.get_BestIndicator(Desired)
    out = best.output()
    outplot.set_ydata(out-out[0])
    plt.pause(0.0001)

#the best weights!
print(
    best.W
)
plt.legend()
plt.show()
