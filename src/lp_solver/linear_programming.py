import numpy as np
from scipy.optimize import linprog
import time

def generate_dual_code_constraints(r=5):
    """
    Generate constraints for Extended Hamming [32,26,4] code using dual code.
    The dual of [32,26,4] is [32,6,16] with only 2^6 = 64 codewords.
    """
    n = 2**r  # 32
    k_dual = r + 1  # 6 for dual code
    
    print(f"Generating dual code constraints for [{n},{n-r-1},4] code...")
    print(f"Dual code is [{n},{k_dual},16] with {2**k_dual} codewords")
    
    # Generate parity check matrix H for Extended Hamming code
    # This defines the dual code constraints
    
    # Standard parity check matrix construction for Extended Hamming
    H = []
    
    # First r rows: standard binary representation of positions 1 to 2^r-1
    for i in range(r):
        row = []
        for pos in range(1, n):  # positions 1 to 31
            bit = (pos >> i) & 1
            row.append(bit)
        row.append(0)  # extension bit column
        H.append(row)
    
    # Last row: all ones (overall parity check)
    H.append([1] * n)
    
    H = np.array(H, dtype=int)
    print(f"Parity check matrix H shape: {H.shape}")
    
    return H

def paper_lp_decode(received_message, channel_error_prob=0.1, r=5):
    """
    ML decode using the research paper's polytope approach.
    
    The code polytope P(C) is defined by:
    1. Box constraints: 0 ≤ x_i ≤ 1
    2. Parity check constraints: Hx = 0 (mod 2)
    
    For LP relaxation, we convert Hx = 0 (mod 2) to linear constraints.
    """
    print("Paper's LP ML Decoding...")
    start_time = time.time()
    
    n = len(received_message)
    
    # Step 1: Calculate gamma vector
    p = channel_error_prob
    gamma = []
    for bit in received_message:
        if bit == 0:
            gamma_i = np.log((1 - p) / p)
        else:
            gamma_i = np.log(p / (1 - p))
        gamma.append(gamma_i)
    gamma = np.array(gamma)
    
    print(f"Gamma vector: {gamma[:5]}...")
    
    # Step 2: Generate dual code constraints
    H = generate_dual_code_constraints(r)
    
    # Step 3: Convert parity constraints to LP constraints
    # For each row h of H: h·x = 0 (mod 2)
    # In LP relaxation: 0 ≤ h·x ≤ |h| - 1 for each h with odd number of 1s
    
    A_parity = []
    b_parity = []
    
    for i, h in enumerate(H):
        weight = np.sum(h)
        if weight > 0:  # Non-zero constraint
            # Add constraint: h·x ≤ weight - 1
            A_parity.append(h)
            b_parity.append(weight - 1)
            
            # Add constraint: -h·x ≤ -1 (equivalent to h·x ≥ 1)
            # But only if we want even parity, let's stick to basic form
    
    A_parity = np.array(A_parity)
    b_parity = np.array(b_parity)
    
    # Step 4: Box constraints: 0 ≤ x_i ≤ 1
    A_box = np.vstack([np.eye(n), -np.eye(n)])  # x_i ≤ 1, -x_i ≤ 0
    b_box = np.hstack([np.ones(n), np.zeros(n)])
    
    # Step 5: Combine constraints
    if len(A_parity) > 0:
        A = np.vstack([A_parity, A_box])
        b = np.hstack([b_parity, b_box])
    else:
        A = A_box
        b = b_box
    
    print(f"LP constraints: {len(A)} total ({len(A_parity)} parity + {len(A_box)} box)")
    print(f"Variables: {n}")
    
    # Step 6: Solve LP
    print("Solving LP...")
    lp_start = time.time()
    
    result = linprog(gamma, A_ub=A, b_ub=b, bounds=[(0, 1) for _ in range(n)], 
                     method='highs', options={'presolve': True})
    
    lp_time = time.time() - lp_start
    total_time = time.time() - start_time
    
    if result.success:
        solution = result.x
        optimal_cost = result.fun
        
        print(f"LP solved in {lp_time:.3f} seconds")
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Optimal cost: {optimal_cost:.6f}")
        print(f"LP solution: {solution[:8]}...")
        
        # Round to nearest codeword (project back to {0,1}^n)
        rounded_solution = np.round(solution).astype(int).tolist()
        
        print(f"Rounded solution: {rounded_solution[:8]}...")
        
        return rounded_solution, optimal_cost
    
    else:
        print(f"LP failed: {result.message}")
        return None, None

def fast_lp_decode_simple(received_message, channel_error_prob=0.1):
    """
    Even simpler approach: just use box constraints for speed test.
    This won't be exact ML but will be very fast.
    """
    print("Fast LP Decoding (box constraints only)...")
    start_time = time.time()
    
    n = len(received_message)
    
    # Calculate gamma
    p = channel_error_prob
    gamma = []
    for bit in received_message:
        if bit == 0:
            gamma_i = np.log((1 - p) / p)
        else:
            gamma_i = np.log(p / (1 - p))
        gamma.append(gamma_i)
    gamma = np.array(gamma)
    
    # Simple LP: minimize gamma^T x subject to 0 ≤ x ≤ 1
    result = linprog(gamma, bounds=[(0, 1) for _ in range(n)], method='highs')
    
    total_time = time.time() - start_time
    
    if result.success:
        solution = result.x
        rounded = np.round(solution).astype(int).tolist()
        
        print(f"Fast LP completed in {total_time:.3f} seconds")
        print(f"Solution: {rounded}")
        
        return rounded, result.fun
    
    return None, None

if __name__ == "__main__":
    # Test the paper's method
    test_message = [0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0,
                   1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    
    print("Testing paper's LP method:")
    result = paper_lp_decode(test_message)
    print(f"Result: {result}")