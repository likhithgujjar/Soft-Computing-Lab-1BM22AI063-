import numpy as np

# Distance matrix representing distances between 3 cities
# np.inf is used to represent no self-loops (infinite cost to stay in the same city)
dist = np.array([[np.inf, 2, 9],
                 [1, np.inf, 6],
                 [15, 7, np.inf]])

# Initialize pheromone matrix with ones (equal pheromone on all paths initially)
pher = np.ones_like(dist)

# Variables to store the best path and its cost found so far
best_path, best_cost = None, float('inf')

# Repeat the process for 20 iterations (simulating 20 ants)
for _ in range(20):
    path = [0]          # Start from city 0
    nodes = {1, 2}      # Cities left to visit

    # Construct a tour by probabilistically choosing the next city
    while nodes:
        cur = path[-1]  # Current city (last visited)

        # Calculate probability of moving to each unvisited city
        # Based on pheromone level and inverse of distance (desirability)
        probs = [(pher[cur, j]) * (1 / dist[cur, j]) for j in nodes]
        probs = np.array(probs) / sum(probs)  # Normalize probabilities

        # Randomly choose the next city based on calculated probabilities
        nxt = np.random.choice(list(nodes), p=probs)
        path.append(nxt)      # Add next city to path
        nodes.remove(nxt)     # Mark it as visited

    path.append(0)  # Return to the starting city to complete the tour

    # Calculate the total cost (distance) of the current path
    cost = sum(dist[path[i], path[i + 1]] for i in range(len(path) - 1))

    # If this path is better than the best so far, store it
    if cost < best_cost:
        best_cost, best_path = cost, path

    # Evaporate pheromone on all paths to simulate natural decay
    pher *= 0.5

    # Deposit pheromone on the edges used in this path
    # More pheromone is added for shorter (better) paths
    for i in range(len(path) - 1):
        pher[path[i], path[i + 1]] += 100 / cost

# Print the best path and its cost after all iterations
print("Best path:", best_path)
print("Min cost:", best_cost)
