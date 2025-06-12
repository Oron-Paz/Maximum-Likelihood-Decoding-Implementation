import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import itertools

def lp_decode(received_message, codewords_list, channel_error_prob=0.1, relaxation='exact'): #defaults to exact LP with no relaxations
    received = np.array(received_message)
    n = len(received)
    
    gamma = compute_gamma(received, channel_error_prob)
    print(f"Gamma (objective): {gamma}")
    
    # STEP 2: Choose relaxation strategy right now this is all i got cus claude sucks at doing the relaxations
    # add more if statements for different methods suggested in the paper
    if relaxation == 'exact':
        return lp_decode_exact(received, codewords_list, gamma)
    else:
        raise ValueError(f"Unknown relaxation: {relaxation}")

def lp_decode_exact(received, codewords_list, gamma):
    """
    EXACT: Use ConvexHull to get true polytope constraints
    Only feasible for small codes due to constraint explosion
    """
    print("Using EXACT polytope (ConvexHull)")
    
    codewords = np.array(codewords_list)
    
    hull = ConvexHull(codewords) #convexHull from scipy returns equations that define the polytopes shapes used bellow
    A_ub = hull.equations[:, :-1]  # Normal vectors
    b_ub = -hull.equations[:, -1]  # Constants
    
    print(f"ConvexHull: {len(A_ub)} facet constraints")
    
    # Solve LP: min γᵀx subject to Ax ≤ b
    # https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html 
    #check out above link for info on whats happening bellow, but baisicly A_ub = coefficents of constraints as a matrix
    # b_ub = the actual value we're constrained on as a vector to match each row in A

    result = linprog(method='highs', c=gamma, A_ub=A_ub, b_ub=b_ub)

    if not result.success:
        print(f"LP failed: {result.message}")
        return None, float('inf')
    
    print(f"\nLP solution x: {result.x}")
    print(f"LP optimal cost: {result.fun:.6f}")
    
    # Find closest codeword to LP solution
    distances = np.sum((codewords - result.x)**2, axis=1)
    closest_idx = np.argmin(distances)
    best_codeword = codewords[closest_idx]
    
    print(f"LP solution distance to codeword: {np.sqrt(distances[closest_idx]):.6f}")
    print(f"Closest codeword: {best_codeword}")
    
    return best_codeword.tolist(), result.fun


def compute_gamma(received, crossover_prob):
    """
    Compute gamma vector for Binary Symmetric Channel
    γᵢ = log(Pr[yᵢ|xᵢ=0] / Pr[yᵢ|xᵢ=1])
    
    For BSC(p): 
    - If received[i] = 0: γᵢ = log((1-p)/p) - prefer xᵢ=0
    - If received[i] = 1: γᵢ = log(p/(1-p)) - prefer xᵢ=1
    
    The sign should make the LP prefer matching the received bit.
    """
    received_array = np.array(received)
    p = crossover_prob
    
    # Avoid log(0) issues
    p = max(min(p, 0.999), 0.001)
    
    gamma = np.zeros(len(received_array))
    
    for i in range(len(received_array)):
        if received_array[i] == 0:
            
            gamma[i] = -np.log((1-p) / p)  
        else:  
            gamma[i] = -np.log(p / (1-p))  
    
    return gamma
