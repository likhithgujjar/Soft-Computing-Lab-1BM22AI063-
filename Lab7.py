import random

def f(x): return x*x

wolves = [random.uniform(-10, 10) for _ in range(5)]
a = 2

for iter in range(10):
    wolves.sort(key=f)
    alpha, beta, delta = wolves[0], wolves[1], wolves[2]
    a -= 2 / 10
    new_wolves = []
    for w in wolves:
        r1, r2 = random.random(), random.random()
        A = 2 * a * r1 - a
        C = 2 * r2
        D_alpha = abs(C * alpha - w)
        X1 = alpha - A * D_alpha

        r1, r2 = random.random(), random.random()
        A = 2 * a * r1 - a
        C = 2 * r2
        D_beta = abs(C * beta - w)
        X2 = beta - A * D_beta

        r1, r2 = random.random(), random.random()
        A = 2 * a * r1 - a
        C = 2 * r2
        D_delta = abs(C * delta - w)
        X3 = delta - A * D_delta

        new_wolves.append((X1 + X2 + X3) / 3)
    wolves = new_wolves
    print(f"Iter {iter+1}: Best = {alpha:.4f}, Value = {f(alpha):.4f}")

print(f"Best solution: {alpha:.4f} with value: {f(alpha):.4f}")
