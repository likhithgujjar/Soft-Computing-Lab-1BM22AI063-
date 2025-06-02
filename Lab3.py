import random

# --- Parameters ---
POP_SIZE = 6       # Number of individuals
GENES = 5          # Length of binary string (enough for 0–31)
GENERATIONS = 10
MUTATION_RATE = 0.1

# --- Fitness Function: f(x) = x^2 ---
def fitness(x):
    return x**2

# --- Generate Initial Population ---
def generate_individual():
    return ''.join(random.choice('01') for _ in range(GENES))

def decode(individual):
    return int(individual, 2)

# --- Selection ---
def selection(population):
    population.sort(key=lambda x: fitness(decode(x)), reverse=True)
    return population[:2]  # select top 2

# --- Crossover ---
def crossover(parent1, parent2):
    point = random.randint(1, GENES - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2

# --- Mutation ---
def mutate(individual):
    individual = list(individual)
    for i in range(GENES):
        if random.random() < MUTATION_RATE:
            individual[i] = '1' if individual[i] == '0' else '0'
    return ''.join(individual)

# --- Main Genetic Algorithm ---
population = [generate_individual() for _ in range(POP_SIZE)]
print("Initial Population:", population)

for generation in range(GENERATIONS):
    print(f"\nGeneration {generation + 1}")
    for ind in population:
        print(f"{ind} => x={decode(ind)}, f(x)={fitness(decode(ind))}")

    # Selection
    parents = selection(population)
    print("Parents:", parents)

    # Create new population
    new_population = parents.copy()
    while len(new_population) < POP_SIZE:
        child1, child2 = crossover(parents[0], parents[1])
        new_population.append(mutate(child1))
        if len(new_population) < POP_SIZE:
            new_population.append(mutate(child2))

    population = new_population

# --- Final Result ---
best = max(population, key=lambda x: fitness(decode(x)))
print(f"\nBest solution: {best} => x={decode(best)}, f(x)={fitness(decode(best))}")
