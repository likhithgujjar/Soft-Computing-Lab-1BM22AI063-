import numpy as np

# --- Fuzzy Sets and Relations ---

# Fuzzy set A (input)
A = np.array([0.3, 0.7, 1.0])

# Fuzzy set B (output)
B = np.array([0.6, 0.2, 0.9])

# Fuzzy relation R: relation from A to intermediate set Y
R = np.random.rand(len(A), 3)

# Fuzzy relation S: relation from intermediate set Y to B
S = np.random.rand(3, len(B))

# --- Max-Min Composition Function ---
def max_min_composition(X, Y):
    result = np.zeros((X.shape[0], Y.shape[1]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[1]):
          result[i][j] = np.max(np.minimum(X[i, :], Y[:, j]))
    return result

# --- Fuzzy Implication (Mamdani) ---
def fuzzy_implication(A, B):
  implication = np.zeros((len(A), len(B)))
  for i in range(len(A)):
      for j in range(len(B)):
          implication[i][j] = min(A[i], B[j])
  return implication

# --- Fuzzy Inference Using Max-Min Composition ---
def fuzzy_inference(input_set, rule_matrix):
    output = np.zeros(rule_matrix.shape[1])
    for j in range(rule_matrix.shape[1]):
        output[j] = np.max(np.minimum(input_set, rule_matrix[:, j]))
    return output

# --- Execution ---

print("Fuzzy Set A:", A)
print("Fuzzy Set B:", B)

# Max-Min Composition: R o S
composition_RS = max_min_composition(R, S)
print("\nMax-Min Composition (R o S):\n", composition_RS)

# Fuzzy Implication Matrix
implication_matrix = fuzzy_implication(A, B)
print("\nFuzzy Implication Matrix (Mamdani):\n", implication_matrix)

# Fuzzy Inference using Implication Matrix
inference_result = fuzzy_inference(A, implication_matrix)
print("\nFuzzy Inference Result:\n", inference_result)
