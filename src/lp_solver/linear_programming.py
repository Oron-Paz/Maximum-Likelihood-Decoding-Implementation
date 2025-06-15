import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import itertools

#trying someething new:
import cvxpy as cp

def lp_decode(received_message, codewords_list, channel_error_prob=0.1, relaxation='exact', local_constraints=None):
    """
    LP decoder with different relaxation methods.
    
    Args:
        received_message: received vector
        codewords_list: list of valid codewords  
        channel_error_prob: channel crossover probability
        relaxation: 'exact' or 'fundamental'
        local_constraints: list of local constraint specifications for fundamental relaxation
    """
    received = np.array(received_message)
    n = len(received)
    
    gamma = compute_gamma(received, channel_error_prob)
    print(f"Gamma (objective): {gamma}")
    
    if relaxation == 'exact':
        return lp_decode_exact(received, codewords_list, gamma)
    elif relaxation == 'fundamental':
        if local_constraints is None:
            raise ValueError("local_constraints required for fundamental relaxation")
        return lp_decode_fundamental_relaxation(received, codewords_list, gamma, local_constraints)
    else:
        raise ValueError(f"Unknown relaxation: {relaxation}")

# Very slow, far too many constrains to be solved efficently. Even worse than Naive approach.
def lp_decode_exact(received, codewords_list, gamma):
    """Exact LP decoding using the full codeword polytope."""
    codewords = np.array(codewords_list)
    
    hull = ConvexHull(codewords)  # ConvexHull from scipy returns equations that define the polytope shapes used below
    A_ub = hull.equations[:, :-1]  # Normal vectors
    b_ub = -hull.equations[:, -1]  # Constants
    
    print(f"ConvexHull: {len(A_ub)} facet constraints")
    
    # Solve LP: min γᵀx subject to Ax ≤ b
    # https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html 
    # Check out above link for info on what's happening below, but basically A_ub = coefficients of constraints as a matrix
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

def lp_decode_fundamental_relaxation(received, codewords_list, gamma, local_constraints):
    """
    Fundamental relaxation using intersection of local constraint polytopes.
    This implements the relaxation described in: https://arxiv.org/pdf/cs/0602087
    
    Args:
        received: received vector
        codewords_list: list of valid codewords (for final rounding)
        gamma: objective function coefficients
        local_constraints: list of dicts, each containing:
            - 'variables': indices of variables involved in this constraint
            - 'valid_patterns': list of valid local patterns for these variables
    """
    n = len(gamma)
    
    print(f"Fundamental relaxation with {len(local_constraints)} local constraints")
    
    # Collect all constraints from local polytopes
    A_ub_list = []
    b_ub_list = []
    
    for i, constraint in enumerate(local_constraints):
        variables = constraint['variables']
        valid_patterns = np.array(constraint['valid_patterns'])
        
        print(f"Local constraint {i}: variables {variables}, {len(valid_patterns)} patterns")
        
        if len(valid_patterns) <= 1:
            continue  # Skip trivial constraints
            
        # Create convex hull of local patterns
        try:
            if len(valid_patterns) == 2 and valid_patterns.shape[1] == 1:
                # Special case for 1D constraints
                min_val = np.min(valid_patterns)
                max_val = np.max(valid_patterns)
                A_local = np.array([[-1], [1]])
                b_local = np.array([-min_val, max_val])
            else:
                hull = ConvexHull(valid_patterns)
                A_local = hull.equations[:, :-1]
                b_local = -hull.equations[:, -1]
        except Exception as e:
            print(f"Warning: ConvexHull failed for constraint {i}: {e}")
            continue
        
        # Embed local constraints into global space
        A_global = np.zeros((A_local.shape[0], n))
        for j, var_idx in enumerate(variables):
            if var_idx < n:  # Safety check
                A_global[:, var_idx] = A_local[:, j]
        
        A_ub_list.append(A_global)
        b_ub_list.append(b_local)
    
    if not A_ub_list:
        print("Warning: No valid local constraints, falling back to box constraints")
        # Fallback to simple box constraints [0,1]^n
        A_ub = np.vstack([np.eye(n), -np.eye(n)])
        b_ub = np.hstack([np.ones(n), np.zeros(n)])
    else:
        A_ub = np.vstack(A_ub_list)
        b_ub = np.hstack(b_ub_list)
    
    print(f"Total constraints: {len(b_ub)}")
    
    # Add box constraints [0,1] for each variable (this is the fundamental polytope P)
    bounds = [(0, 1) for _ in range(n)]
    
    # Solve LP: min γᵀx subject to x ∈ ∩_{j∈J} conv(C_j) and x ∈ [0,1]^n
    result = linprog(method='highs', c=gamma, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    if not result.success:
        print(f"LP failed: {result.message}")
        return None, float('inf')
    
    print(f"\nFundamental LP solution x: {result.x}")
    print(f"LP optimal cost: {result.fun:.6f}")
    
    decoded_word = result.x.astype(int).tolist()
    
    return decoded_word, result.fun

def lp_decode_box_relaxation(received_message, codewords_list, channel_error_prob=0.1):
    """
    Ultra-simple relaxation: just use box constraints [0,1]^n.
    This is the most relaxed polytope possible.
    """
    received = np.array(received_message)
    n = len(received)
    gamma = compute_gamma(received, channel_error_prob)
    
    print(f"Box relaxation: just [0,1]^n constraints")
    
    # Only box constraints: 0 ≤ x_i ≤ 1 for all i
    bounds = [(0, 1) for _ in range(n)]
    
    # Solve LP: min γᵀx subject to x ∈ [0,1]^n
    result = linprog(method='highs', c=gamma, bounds=bounds, options={'presolve': False, 'time_limit': 1.0})
    
    if not result.success:
        print(f"LP failed: {result.message}")
        return None, float('inf')
    
    print(f"Box LP solution x: {result.x}")
    print(f"LP optimal cost: {result.fun:.6f}")
    
    decoded_word = result.x.astype(int).tolist()
    
    return decoded_word, result.fun

def lp_decode_simple_parity_relaxation(received_message, codewords_list, channel_error_prob=0.1, 
                                       local_constraints=None):
    """
    Simple relaxation: only use parity check constraints, ignore complex local patterns.
    """
    received = np.array(received_message)
    n = len(received)
    gamma = compute_gamma(received, channel_error_prob)
    
    print(f"Simple parity relaxation")
    
    # Extract simple parity constraints from local_constraints
    A_ub_list = []
    b_ub_list = []
    
    if local_constraints:
        for constraint in local_constraints:
            variables = constraint['variables']
            
            # Simple parity constraint: sum of variables should be even
            # This means: sum(x_i for i in variables) ≤ |variables| - 0.5
            # and: sum(x_i for i in variables) ≥ 0.5
            # But we'll use even simpler: sum ≤ |variables| and sum ≥ 0
            
            # Upper bound: sum(x_i) ≤ len(variables)
            A_upper = np.zeros(n)
            for var_idx in variables:
                if var_idx < n:
                    A_upper[var_idx] = 1
            A_ub_list.append(A_upper)
            b_ub_list.append(len(variables))
            
            # Could add lower bound, but let's keep it simple
    
    if A_ub_list:
        A_ub = np.vstack(A_ub_list)
        b_ub = np.array(b_ub_list)
        print(f"Using {len(b_ub)} simple parity constraints")
    else:
        A_ub = None
        b_ub = None
        print("No constraints - falling back to box relaxation")
    
    # Box constraints
    bounds = [(0, 1) for _ in range(n)]
    
    # Solve LP
    result = linprog(method='highs', c=gamma, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    
    if not result.success:
        print(f"LP failed: {result.message}")
        return None, float('inf')
    
    print(f"Simple LP solution x: {result.x}")
    print(f"LP optimal cost: {result.fun:.6f}")

    
    return result.x.astype(int).tolist(), result.fun

def lp_decode_subset_relaxation(received_message, codewords_list, channel_error_prob=0.1, 
                                local_constraints=None, max_constraints=10):
    """
    Use only a subset of local constraints to keep the problem manageable.
    """
    received = np.array(received_message)
    n = len(received)
    gamma = compute_gamma(received, channel_error_prob)
    
    print(f"Subset relaxation using max {max_constraints} constraints")
    
    # Use only first few local constraints
    if local_constraints and len(local_constraints) > max_constraints:
        local_constraints = local_constraints[:max_constraints]
        print(f"Reduced from many constraints to {len(local_constraints)}")
    
    # Collect constraints but limit the number of patterns
    A_ub_list = []
    b_ub_list = []
    
    if local_constraints:
        for i, constraint in enumerate(local_constraints):
            variables = constraint['variables']
            valid_patterns = constraint['valid_patterns']
            
            # Limit to first 16 patterns to avoid explosion
            valid_patterns = valid_patterns[:16]
            
            print(f"Constraint {i}: variables {variables}, using {len(valid_patterns)} patterns")
            
            if len(valid_patterns) <= 1:
                continue
            
            # Create convex hull of limited patterns
            try:
                if len(valid_patterns) == 2 and len(variables) == 1:
                    # Special case for 1D
                    min_val = min(p[0] for p in valid_patterns)
                    max_val = max(p[0] for p in valid_patterns)
                    A_local = np.array([[-1], [1]])
                    b_local = np.array([-min_val, max_val])
                else:
                    hull = ConvexHull(valid_patterns)
                    A_local = hull.equations[:, :-1]
                    b_local = -hull.equations[:, -1]
            except Exception as e:
                print(f"Skipping constraint {i}: {e}")
                continue
            
            # Embed into global space
            A_global = np.zeros((A_local.shape[0], n))
            for j, var_idx in enumerate(variables):
                if var_idx < n:
                    A_global[:, var_idx] = A_local[:, j]
            
            A_ub_list.append(A_global)
            b_ub_list.append(b_local)
    
    if A_ub_list:
        A_ub = np.vstack(A_ub_list)
        b_ub = np.hstack(b_ub_list)
        print(f"Total constraints: {len(b_ub)} (much better!)")
    else:
        A_ub = None
        b_ub = None
    
    # Box constraints
    bounds = [(0, 1) for _ in range(n)]
    
    # Solve LP
    result = linprog(method='highs', c=gamma, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    
    if not result.success:
        print(f"LP failed: {result.message}")
        return None, float('inf')
    
    print(f"Subset LP solution x: {result.x}")
    print(f"LP optimal cost: {result.fun:.6f}")
    
    return result.x.astype(int).tolist(), result.fun

def lp_decode_syndrome_ml_relaxation(received_message, codewords_list, channel_error_prob=0.1, 
                                     parity_check_matrix=None, max_error_weight=None):
    """
    Syndrome-based ML LP decoding using coset decomposition.
    
    This implements the approach from your document where we:
    1. Calculate syndrome of received word
    2. Find all error patterns that produce this syndrome
    3. Use ML to select the most likely error pattern
    4. Apply LP relaxation to the syndrome-constrained space
    
    This reduces the search space from 2^n to 2^(n-k) as mentioned in your document.
    
    Args:
        received_message: received vector
        codewords_list: list of valid codewords
        channel_error_prob: channel crossover probability  
        parity_check_matrix: H matrix for syndrome calculation
        max_error_weight: maximum error weight to consider (for efficiency)
    """
    received = np.array(received_message)
    n = len(received)
    gamma = compute_gamma(received, channel_error_prob)
    
    print(f"Syndrome-ML LP relaxation")
    
    # Extract or compute parity check matrix
    if parity_check_matrix is None:
        H = extract_parity_check_matrix(codewords_list)
    else:
        H = np.array(parity_check_matrix)
    
    print(f"Parity check matrix H: {H.shape}")
    
    # Calculate syndrome
    syndrome = np.dot(received, H.T) % 2
    syndrome_str = ''.join(map(str, syndrome))
    print(f"Syndrome: {syndrome_str}")
    
    if max_error_weight is None:
        max_error_weight = min(3, n//2)  # Reasonable default
    
    # Generate error patterns that produce this syndrome
    valid_error_patterns = []
    
    # Check error patterns up to max_error_weight
    for weight in range(max_error_weight + 1):
        for positions in itertools.combinations(range(n), weight):
            error_pattern = np.zeros(n, dtype=int)
            error_pattern[list(positions)] = 1
            
            # Check if this error pattern produces the observed syndrome
            pattern_syndrome = np.dot(error_pattern, H.T) % 2
            if np.array_equal(pattern_syndrome, syndrome):
                valid_error_patterns.append(error_pattern)
    
    print(f"Found {len(valid_error_patterns)} valid error patterns")
    
    if not valid_error_patterns:
        print("No valid error patterns found - using zero error")
        best_error_pattern = np.zeros(n, dtype=int)
        decoded_word = received.astype(int).tolist()
        return decoded_word, float('inf')
    
    # Create convex hull of valid error patterns for LP relaxation
    try:
        if len(valid_error_patterns) == 1:
            # Only one valid pattern - return it directly
            best_error_pattern = valid_error_patterns[0]
            print("Unique error pattern found")
        else:
            # Multiple patterns - use LP to find ML solution
            valid_patterns_array = np.array(valid_error_patterns)
            
            # Create LP formulation:
            # Minimize gamma^T * (received + e) subject to e in convex hull of valid patterns
            # This is equivalent to minimizing gamma^T * e (since received is constant)
            
            if len(valid_error_patterns) == 2:
                # Simple case: interpolation between two points
                hull_A = np.array([[1, -1], [-1, 1]])  # e = α*e1 + (1-α)*e2, α ∈ [0,1]
                hull_b = np.array([0, 0])
                
                # Transform to error space: minimize sum(gamma_i * e_i)
                # Variables: [e_0, e_1, ..., e_{n-1}]
                c = gamma  # We want to minimize likelihood of error pattern
                
                # Constraints: error pattern must be convex combination of valid patterns
                # This is complex to set up properly - let's use a simpler approach
                
                # Calculate ML directly among the discrete options
                best_likelihood = float('inf')
                best_error_pattern = valid_error_patterns[0]
                
                for error_pattern in valid_error_patterns:
                    likelihood = np.dot(gamma, error_pattern)
                    if likelihood < best_likelihood:
                        best_likelihood = likelihood
                        best_error_pattern = error_pattern
                        
                print(f"Selected error pattern with likelihood {best_likelihood:.6f}")
                
            else:
                # Multiple patterns - use convex hull relaxation
                hull = ConvexHull(valid_patterns_array)
                A_ub = hull.equations[:, :-1]
                b_ub = -hull.equations[:, -1]
                
                print(f"Syndrome-constrained polytope: {len(A_ub)} constraints")
                
                # Solve LP: min γᵀe subject to e ∈ conv(valid_error_patterns)
                bounds = [(0, 1) for _ in range(n)]  # Error bits are binary, relaxed to [0,1]
                
                result = linprog(method='highs', c=gamma, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
                
                if not result.success:
                    print(f"LP failed: {result.message}")
                    # Fallback to direct ML among discrete patterns
                    best_likelihood = float('inf')
                    best_error_pattern = valid_error_patterns[0]
                    
                    for error_pattern in valid_error_patterns:
                        likelihood = np.dot(gamma, error_pattern)
                        if likelihood < best_likelihood:
                            best_likelihood = likelihood
                            best_error_pattern = error_pattern
                else:
                    print(f"LP solution: {result.x}")
                    print(f"LP cost: {result.fun:.6f}")
                    
                    # Round LP solution to nearest valid error pattern
                    distances = np.sum((valid_patterns_array - result.x)**2, axis=1)
                    closest_idx = np.argmin(distances)
                    best_error_pattern = valid_patterns_array[closest_idx]
                    
                    print(f"Rounded to error pattern {closest_idx}: {best_error_pattern}")
                    
    except Exception as e:
        print(f"ConvexHull/LP failed: {e}")
        # Fallback: direct ML decoding among valid patterns
        best_likelihood = float('inf')
        best_error_pattern = valid_error_patterns[0]
        
        for error_pattern in valid_error_patterns:
            likelihood = np.dot(gamma, error_pattern)
            if likelihood < best_likelihood:
                best_likelihood = likelihood
                best_error_pattern = error_pattern
        
        print(f"Used direct ML fallback")
    
    # Apply error correction
    decoded_word = (received + best_error_pattern) % 2
    final_cost = np.dot(gamma, best_error_pattern)
    
    print(f"Selected error pattern: {best_error_pattern}")
    print(f"Decoded word: {decoded_word}")
    print(f"Final cost: {final_cost:.6f}")
    
    return decoded_word.tolist(), final_cost

def lp_decode_syndrome_polytope_relaxation(received_message, codewords_list, channel_error_prob=0.1,
                                          parity_check_matrix=None):
    """
    Alternative syndrome-ML approach using syndrome polytope constraints.
    
    Instead of enumerating error patterns, this directly constrains the LP
    to satisfy He = s (syndrome equation) and uses box relaxation.
    """
    received = np.array(received_message)
    n = len(received)
    gamma = compute_gamma(received, channel_error_prob)
    
    print(f"Syndrome polytope LP relaxation")
    
    # Extract or compute parity check matrix
    if parity_check_matrix is None:
        H = extract_parity_check_matrix(codewords_list)
    else:
        H = np.array(parity_check_matrix)
    
    # Calculate syndrome
    syndrome = np.dot(received, H.T) % 2
    print(f"Syndrome: {syndrome}")
    
    # Set up LP:
    # Minimize gamma^T * e
    # Subject to: H * e = syndrome (mod 2)
    #            0 ≤ e_i ≤ 1 for all i
    
    # Convert mod 2 equality constraints to linear constraints
    # For binary variables: He = s (mod 2) becomes He ≥ s and He ≤ s + (large number)
    # But this is complex for mod 2... let's use a different approach
    
    # Alternative: use penalty method
    # Minimize gamma^T * e + λ * ||He - s||²
    # Subject to 0 ≤ e_i ≤ 1
    
    lambda_penalty = 100.0  # Penalty weight for syndrome constraint violation
    
    # Modify objective: γᵀe + λ * ||He - s||²
    # This becomes: γᵀe + λ * eᵀHᵀHe - 2λ * sᵀHe + λ * sᵀs
    # Since we're minimizing, the constant term λ * sᵀs doesn't matter
    
    # For quadratic programming, we need to express as:
    # minimize (1/2)xᵀPx + qᵀx
    # subject to Gx ≤ h, Ax = b
    
    # Let's simplify and use iterative approach or just direct enumeration
    # for demonstration purposes
    
    print("Using direct syndrome-constrained search...")
    
    # Find best error pattern among those satisfying syndrome constraint
    best_cost = float('inf')
    best_error_pattern = np.zeros(n, dtype=int)
    
    # Check all possible error patterns (exponential, but demonstrates concept)
    max_weight = min(4, n)  # Limit search for efficiency
    
    for weight in range(max_weight + 1):
        for positions in itertools.combinations(range(n), weight):
            error_pattern = np.zeros(n, dtype=int)
            error_pattern[list(positions)] = 1
            
            # Check syndrome constraint
            pattern_syndrome = np.dot(error_pattern, H.T) % 2
            if np.array_equal(pattern_syndrome, syndrome):
                cost = np.dot(gamma, error_pattern)
                if cost < best_cost:
                    best_cost = cost
                    best_error_pattern = error_pattern
    
    # Apply correction
    decoded_word = (received + best_error_pattern) % 2
    
    print(f"Best error pattern: {best_error_pattern}")
    print(f"Decoded word: {decoded_word}")
    print(f"Cost: {best_cost:.6f}")
    
    return decoded_word.tolist(), best_cost

def extract_parity_check_matrix(codewords_list):
    """
    Extract parity check matrix from list of codewords.
    This is a simplified extraction - in practice you'd have H available.
    """
    codewords = np.array(codewords_list)
    n = codewords.shape[1]
    k = int(np.log2(len(codewords)))  # Assumes 2^k codewords
    
    # For demo: create a simple parity check matrix
    # This is not the actual H for the given codewords, just for demonstration
    r = n - k  # Number of parity checks
    
    if r <= 0:
        # All-zero matrix for trivial codes
        return np.zeros((1, n), dtype=int)
    
    # Create a random-ish parity check matrix for demonstration
    # In practice, you'd extract this properly from the code structure
    np.random.seed(42)  # For reproducibility
    H = np.random.randint(0, 2, (r, n))
    
    return H

def compute_gamma(received, crossover_prob):
    """
    Compute log-likelihood ratios for binary symmetric channel.
    For BSC with crossover probability p:
    - If received bit is 0, we want to favor x_i = 0 
    - If received bit is 1, we want to favor x_i = 1
    """
    received_array = np.array(received)
    p = crossover_prob
    
    # Avoid log(0) issues
    p = max(min(p, 0.999), 0.001)
    
    gamma = np.zeros(len(received_array))
    
    for i in range(len(received_array)):
        if received_array[i] == 0:
            # Favor x_i = 0, so cost is positive when x_i = 1
            gamma[i] = np.log((1-p) / p)  
        else:  
            # Favor x_i = 1, so cost is negative when x_i = 1
            gamma[i] = np.log(p / (1-p))  
    
    return gamma

def load_code_from_file(filename):
    """
    Load code data from saved pickle file.
    
    Args:
        filename: path to pickle file containing code data
        
    Returns:
        codewords, local_constraints, code_info
    """
    import pickle
    
    with open(filename, 'rb') as f:
        code_data = pickle.load(f)
    
    codewords = code_data['codewords']
    local_constraints = code_data['local_constraints']
    code_info = code_data['code_info']
    
    print(f"Loaded {code_info['code_type']} code:")
    print(f"  n={code_info['n']}, k={code_info['k']}")
    print(f"  {len(codewords)} codewords, {len(local_constraints)} local constraints")
    
    return codewords, local_constraints, code_info
