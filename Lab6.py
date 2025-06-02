import random

def f(x): return x*x

pos = [random.uniform(-10,10) for _ in range(5)]
vel = [0]*5
pbest = pos[:]
pbest_val = [f(x) for x in pos]
gbest = min(pbest, key=f)

for it in range(10):
    for i in range(5):
        if f(pos[i]) < pbest_val[i]:
            pbest[i] = pos[i]
            pbest_val[i] = f(pos[i])
    gbest = min(pbest, key=f)
    for i in range(5):
        vel[i] = 0.5*vel[i] + 1.5*random.random()*(pbest[i]-pos[i]) + 1.5*random.random()*(gbest-pos[i])
        pos[i] += vel[i]
    print(f"Iteration {it+1}: Best position = {gbest:.4f}, Best value = {f(gbest):.4f}")

print("\nFinal Best position:", gbest)
print("Final Best value:", f(gbest))
