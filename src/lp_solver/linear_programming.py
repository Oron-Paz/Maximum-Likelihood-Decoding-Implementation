import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import itertools

def lp_decode(received_message, codewords_list, channel_error_prob=0.1, relaxation='exact', parity_check_matrix=None): 
    received = np.array(received_message)
    n = len(received)
    
    gamma = compute_gamma(received, channel_error_prob)
    print(f"Gamma (objective): {gamma}")
    
    # add more if statements for different methods suggested in the paper
    if relaxation == 'exact':
        return lp_decode_exact(received, codewords_list, gamma)
    elif relaxation == 'fundamental':
        if parity_check_matrix is None:
            raise ValueError("Parity check matrix required for fundamental polytope relaxation")
        return lp_decode_fundamental_polytope(received, parity_check_matrix, gamma, codewords_list)
    else:
        raise ValueError(f"Unknown relaxation: {relaxation}")

def lp_decode_exact(received, codewords_list, gamma):
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

def lp_decode_fundamental_polytope(received, parity_check_matrix, gamma, codewords_list):
    """
    LP decoder using the fundamental polytope relaxation as described in 
    Feldman-Wainwright-Karger paper. This implements the relaxation shown
    in the PDF images.
    
    The fundamental polytope P is defined by:
    P = {ω ∈ ℝⁿ | ∀i ∈ I : 0 ≤ ωᵢ ≤ 1 and
                  ∀j ∈ J, ∀I'ⱼ ⊆ Iⱼ, |I'ⱼ| odd :
                  ∑ᵢ∈I'ⱼ ωᵢ + ∑ᵢ∈(Iⱼ\I'ⱼ) (1 - ωᵢ) ≤ |Iⱼ| - 1}
    
    Args:
        received: Received vector
        parity_check_matrix: Parity check matrix H
        gamma: Objective function coefficients
        codewords_list: List of codewords (for final decoding step)
    """
    H = np.array(parity_check_matrix)
    m, n = H.shape
    codewords = np.array(codewords_list)
    
    print(f"Building fundamental polytope constraints...")
    print(f"Parity check matrix: {m} × {n}")
    
    # Build constraints for fundamental polytope
    A_ub = []
    b_ub = []
    
    # 1. Box constraints: 0 ≤ ωᵢ ≤ 1
    # Upper bounds: ωᵢ ≤ 1
    A_ub.extend(np.eye(n))
    b_ub.extend([1.0] * n)
    
    # Lower bounds: -ωᵢ ≤ 0 (equivalent to ωᵢ ≥ 0)
    A_ub.extend(-np.eye(n))
    b_ub.extend([0.0] * n)
    
    # 2. Parity check constraints
    # For each parity check j, and each odd-sized subset I'ⱼ of the support Iⱼ:
    # ∑ᵢ∈I'ⱼ ωᵢ + ∑ᵢ∈(Iⱼ\I'ⱼ) (1 - ωᵢ) ≤ |Iⱼ| - 1
    # 
    # This can be rewritten as:
    # ∑ᵢ∈I'ⱼ ωᵢ + ∑ᵢ∈(Iⱼ\I'ⱼ) (1 - ωᵢ) ≤ |Iⱼ| - 1
    # ∑ᵢ∈I'ⱼ ωᵢ + |Iⱼ\I'ⱼ| - ∑ᵢ∈(Iⱼ\I'ⱼ) ωᵢ ≤ |Iⱼ| - 1
    # ∑ᵢ∈I'ⱼ ωᵢ - ∑ᵢ∈(Iⱼ\I'ⱼ) ωᵢ ≤ |Iⱼ| - 1 - |Iⱼ\I'ⱼ|
    # ∑ᵢ∈I'ⱼ ωᵢ - ∑ᵢ∈(Iⱼ\I'ⱼ) ωᵢ ≤ |I'ⱼ| - 1
    
    constraint_count = 0
    for j in range(m):  # For each parity check
        # Find support of j-th parity check (Iⱼ)
        support_j = np.where(H[j, :] == 1)[0]
        support_size = len(support_j)
        
        if support_size == 0:
            continue
            
        # Generate all odd-sized subsets of the support
        max_subset_size = min(support_size, 15)  # Limit to prevent explosion
        
        for subset_size in range(1, max_subset_size + 1, 2):  # Odd sizes only
            # Generate all subsets of this odd size
            from itertools import combinations
            for subset in combinations(support_j, subset_size):
                I_prime = set(subset)
                I_complement = set(support_j) - I_prime
                
                # Build constraint vector
                constraint_vec = np.zeros(n)
                
                # Coefficients for variables in I'ⱼ: +1
                for i in I_prime:
                    constraint_vec[i] = 1.0
                
                # Coefficients for variables in Iⱼ\I'ⱼ: -1  
                for i in I_complement:
                    constraint_vec[i] = -1.0
                
                # Right-hand side: |I'ⱼ| - 1
                rhs = len(I_prime) - 1
                
                A_ub.append(constraint_vec)
                b_ub.append(rhs)
                constraint_count += 1
                
                # Limit total constraints to prevent memory issues
                if constraint_count > 10000:
                    print(f"  Limiting to {constraint_count} parity constraints")
                    break
            
            if constraint_count > 10000:
                break
        
        if constraint_count > 10000:
            break
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    print(f"Fundamental polytope constraints: {len(A_ub)} total")
    print(f"  Box constraints: {2*n}")
    print(f"  Parity constraints: {constraint_count}")
    
    # Solve LP: min γᵀω subject to A_ub ω ≤ b_ub
    result = linprog(method='highs', c=gamma, A_ub=A_ub, b_ub=b_ub)
    
    if not result.success:
        print(f"LP failed: {result.message}")
        return None, float('inf')
    
    print(f"\nLP solution ω: {result.x}")
    print(f"LP optimal cost: {result.fun:.6f}")
    
    # Check if solution is integral (a codeword)
    solution = result.x
    is_integral = np.allclose(solution, np.round(solution), atol=1e-6)
    
    if is_integral:
        # Round to binary and check if it's a valid codeword
        binary_solution = np.round(solution).astype(int)
        solution_tuple = tuple(binary_solution.tolist())
        codeword_tuples = set(tuple(cw) for cw in codewords_list)
        
        if solution_tuple in codeword_tuples:
            print("LP solution is a valid codeword!")
            return binary_solution.tolist(), result.fun
        else:
            print("LP solution is integral but not a valid codeword (pseudocodeword)")
    else:
        print("LP solution is fractional (pseudocodeword)")
    
    # Find closest valid codeword to LP solution
    distances = np.sum((codewords - solution)**2, axis=1)
    closest_idx = np.argmin(distances)
    best_codeword = codewords[closest_idx]
    
    print(f"Distance to closest codeword: {np.sqrt(distances[closest_idx]):.6f}")
    print(f"Closest codeword: {best_codeword}")
    
    return best_codeword.tolist(), result.fun


def compute_gamma(received, crossover_prob):
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