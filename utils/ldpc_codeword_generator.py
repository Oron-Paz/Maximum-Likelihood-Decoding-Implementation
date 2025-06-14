import numpy as np
import itertools

def generate_ldpc_code(n, k, max_check_degree=3, max_var_degree=2, seed=42):
    """
    Generate a Low-Density Parity-Check (LDPC) code.
    
    Args:
        n: block length (number of bits)
        k: information bits (n-k = number of parity checks)
        max_check_degree: maximum number of variables per check
        max_var_degree: maximum number of checks per variable
        seed: random seed for reproducibility
    
    Returns:
        codewords: list of valid codewords
        local_constraints: list of local constraint specifications
        H: parity check matrix (m x n)
        code_info: dictionary with code parameters
    """
    np.random.seed(seed)
    
    m = n - k  # number of parity checks
    print(f"Generating LDPC code: n={n}, k={k}, m={m}")
    
    # Generate better structured H matrix
    H = generate_better_parity_matrix(n, m, max_check_degree, max_var_degree)
    
    # Convert to systematic form and find actual rank
    H_sys, rank = convert_to_systematic(H)
    actual_k = n - rank
    
    print(f"Matrix rank: {rank}, Actual k: {actual_k}")
    
    # Generate codewords efficiently
    if actual_k <= 18:  # Full enumeration for reasonable sizes
        codewords = generate_all_codewords_exhaustive(H, n, actual_k)
    else:
        # Sample approach for very large codes
        codewords = generate_sample_codewords_improved(H, n, actual_k, num_samples=min(10000, 2**min(actual_k, 15)))
    
    # Remove duplicates
    unique_codewords = []
    seen = set()
    for cw in codewords:
        cw_tuple = tuple(cw)
        if cw_tuple not in seen:
            seen.add(cw_tuple)
            unique_codewords.append(cw)
    
    codewords = unique_codewords
    
    # Create local constraints from parity checks
    local_constraints = create_ldpc_local_constraints(H)
    
    density = np.mean(H)
    code_info = {
        'n': n,
        'k': actual_k,
        'm': rank,
        'designed_k': k,
        'density': density,
        'max_check_degree': max_check_degree,
        'max_var_degree': max_var_degree,
        'num_codewords': len(codewords),
        'num_constraints': len(local_constraints)
    }
    
    print(f"Generated {len(codewords)} unique codewords (expected 2^{actual_k} = {2**actual_k})")
    print(f"Created {len(local_constraints)} local constraints")
    
    return codewords, local_constraints, H, code_info

def generate_better_parity_matrix(n, m, max_check_degree, max_var_degree):
    """Generate a better structured parity check matrix."""
    H = np.zeros((m, n), dtype=int)
    
    # Ensure each check has at least 2 variables
    for i in range(m):
        num_vars = np.random.randint(2, min(max_check_degree + 1, n + 1))
        var_indices = np.random.choice(n, size=num_vars, replace=False)
        H[i, var_indices] = 1
    
    # Balance variable degrees
    for j in range(n):
        current_degree = np.sum(H[:, j])
        if current_degree == 0:
            # Connect to at least one check
            check_idx = np.random.randint(m)
            H[check_idx, j] = 1
        elif current_degree > max_var_degree:
            # Remove excess connections
            check_indices = np.where(H[:, j] == 1)[0]
            excess = current_degree - max_var_degree
            remove_indices = np.random.choice(check_indices, size=excess, replace=False)
            H[remove_indices, j] = 0
    
    return H

def convert_to_systematic(H):
    """Convert parity check matrix to systematic form and find rank."""
    m, n = H.shape
    H_work = H.copy()
    
    # Gaussian elimination in GF(2)
    rank = 0
    for col in range(min(m, n)):
        # Find pivot
        pivot_row = None
        for row in range(rank, m):
            if H_work[row, col] == 1:
                pivot_row = row
                break
        
        if pivot_row is None:
            continue
        
        # Swap rows if needed
        if pivot_row != rank:
            H_work[[rank, pivot_row]] = H_work[[pivot_row, rank]]
        
        # Eliminate other 1s in this column
        for row in range(m):
            if row != rank and H_work[row, col] == 1:
                H_work[row] = (H_work[row] + H_work[rank]) % 2
        
        rank += 1
    
    return H_work, rank

def generate_all_codewords_exhaustive(H, n, k):
    """Generate all valid codewords by exhaustive search."""
    codewords = []
    max_search = min(2**n, 2**20)  # Cap the search space
    
    print(f"Searching up to {max_search} patterns for valid codewords...")
    
    for i in range(max_search):
        # Convert to binary vector
        x = np.array([(i >> j) & 1 for j in range(n)])
        
        # Check if this satisfies H*x = 0 (mod 2)
        syndrome = np.dot(H, x) % 2
        if np.all(syndrome == 0):
            codewords.append(x.tolist())
        
        if len(codewords) >= 2**k:
            break  # Found enough codewords
    
    return codewords

def generate_sample_codewords_improved(H, n, k, num_samples=1000):
    """Generate sample codewords using improved random search."""
    codewords = []
    max_attempts = num_samples * 50
    attempts = 0
    
    print(f"Sampling up to {num_samples} codewords...")
    
    while len(codewords) < num_samples and attempts < max_attempts:
        # Random binary vector
        x = np.random.randint(0, 2, n)
        
        # Check if it satisfies H*x = 0
        syndrome = np.dot(H, x) % 2
        if np.all(syndrome == 0):
            codewords.append(x.tolist())
        
        attempts += 1
        
        if attempts % 5000 == 0:
            print(f"  Found {len(codewords)} codewords after {attempts} attempts")
    
    return codewords

def solve_ldpc_encoding(info_bits, H, n, k, m):
    """
    Solve for a valid codeword given information bits.
    This is a simplified systematic encoder.
    """
    # This function is kept for compatibility but improved implementation
    # is now in the main generate function
    
    # Try systematic approach
    x = np.zeros(n, dtype=int)
    x[:len(info_bits)] = info_bits
    
    # Simple approach: try random parity bits until we find a valid codeword
    for attempt in range(1000):
        # Set random parity bits
        x[len(info_bits):] = np.random.randint(0, 2, n - len(info_bits))
        
        # Check if valid
        syndrome = np.dot(H, x) % 2
        if np.all(syndrome == 0):
            return x
    
    return None

def generate_codewords_brute_force(H, n, max_codewords=16):
    """Brute force search for valid codewords."""
    codewords = []
    max_attempts = min(2**n, 2**18, max_codewords * 1000)
    
    for i in range(max_attempts):
        if len(codewords) >= max_codewords:
            break
            
        # Convert i to binary representation
        x = np.array([(i >> j) & 1 for j in range(n)])
        
        # Check if this satisfies all parity checks
        syndrome = np.dot(H, x) % 2
        if np.all(syndrome == 0):
            codewords.append(x.tolist())
    
    return codewords

def encode_systematic_ldpc(info_bits, H, n, k):
    """
    Encode information bits using systematic LDPC encoding.
    
    Args:
        info_bits: information bits (length k)
        H: parity check matrix
        n: block length
        k: information length
    
    Returns:
        x: encoded codeword or None if encoding fails
    """
    # This is now handled better in the main generation function
    return solve_ldpc_encoding(info_bits, H, n, k, n-k)

def create_ldpc_local_constraints(H):
    """
    Create local constraints from LDPC parity check matrix.
    
    Args:
        H: parity check matrix (m x n)
    
    Returns:
        local_constraints: list of constraint dictionaries
    """
    local_constraints = []
    m, n = H.shape
    
    for i in range(m):
        # Find variables involved in this parity check
        variables = [j for j in range(n) if H[i, j] == 1]
        
        if len(variables) <= 1:
            continue
        
        # Generate all valid patterns for this parity check (even parity)
        valid_patterns = []
        for pattern in itertools.product([0, 1], repeat=len(variables)):
            if sum(pattern) % 2 == 0:  # Even parity constraint
                valid_patterns.append(list(pattern))
        
        if len(valid_patterns) > 1:  # Skip trivial constraints
            local_constraints.append({
                'variables': variables,
                'valid_patterns': valid_patterns,
                'constraint_type': 'parity_check',
                'check_index': i
            })
    
    return local_constraints

def generate_regular_ldpc(n, j, k):
    """
    Generate a regular LDPC code where each variable has degree j and each check has degree k.
    
    Args:
        n: block length
        j: variable node degree (number of checks each variable participates in)
        k: check node degree (number of variables each check involves)
    
    Returns:
        Same as generate_ldpc_code
    """
    # Number of checks
    m = (n * j) // k
    
    if (n * j) % k != 0:
        raise ValueError(f"Cannot create regular LDPC: n*j={n*j} not divisible by k={k}")
    
    print(f"Generating regular LDPC: n={n}, m={m}, variable degree={j}, check degree={k}")
    
    # Create regular bipartite graph
    H = np.zeros((m, n), dtype=int)
    
    # Create edge list
    edges = []
    for var in range(n):
        for _ in range(j):
            edges.append(var)
    
    # Randomly assign edges to checks
    np.random.shuffle(edges)
    
    for check in range(m):
        for i in range(k):
            if check * k + i < len(edges):
                var = edges[check * k + i]
                H[check, var] = 1
    
    # Generate codewords using the same improved method
    info_bits = n - m
    
    # Use the improved generation method
    if info_bits <= 15:
        codewords = generate_all_codewords_exhaustive(H, n, info_bits)
    else:
        codewords = generate_sample_codewords_improved(H, n, info_bits, num_samples=min(1000, 2**min(info_bits, 12)))
    
    local_constraints = create_ldpc_local_constraints(H)
    
    code_info = {
        'n': n,
        'k': info_bits,
        'm': m,
        'variable_degree': j,
        'check_degree': k,
        'regular': True,
        'num_codewords': len(codewords),
        'num_constraints': len(local_constraints)
    }
    
    print(f"Generated {len(codewords)} codewords for regular LDPC")
    
    return codewords, local_constraints, H, code_info

def print_ldpc_info(H, code_info):
    """Print detailed information about the LDPC code."""
    print("\n" + "="*50)
    print("LDPC CODE INFORMATION")
    print("="*50)
    
    print(f"Block length (n): {code_info['n']}")
    print(f"Information bits (k): {code_info['k']}")
    print(f"Parity checks (m): {code_info['m']}")
    print(f"Rate: {code_info['k']/code_info['n']:.3f}")
    print(f"Matrix density: {code_info.get('density', 0):.3f}")
    
    if 'regular' in code_info and code_info['regular']:
        print(f"Regular LDPC:")
        print(f"  Variable degree: {code_info['variable_degree']}")
        print(f"  Check degree: {code_info['check_degree']}")
    else:
        print(f"Irregular LDPC:")
        print(f"  Max check degree: {code_info.get('max_check_degree', 'N/A')}")
        print(f"  Max variable degree: {code_info.get('max_var_degree', 'N/A')}")
    
    print(f"Number of codewords: {code_info['num_codewords']}")
    print(f"Number of local constraints: {code_info['num_constraints']}")
    
    # Degree distribution
    var_degrees = np.sum(H, axis=0)
    check_degrees = np.sum(H, axis=1)
    
    print(f"\nDegree distribution:")
    print(f"  Variable degrees: min={np.min(var_degrees)}, max={np.max(var_degrees)}, avg={np.mean(var_degrees):.2f}")
    print(f"  Check degrees: min={np.min(check_degrees)}, max={np.max(check_degrees)}, avg={np.mean(check_degrees):.2f}")

if __name__ == "__main__":
    # Example usage
    print("Generating small LDPC code for testing...")
    codewords, constraints, H, info = generate_ldpc_code(n=8, k=4, max_check_degree=3)
    print_ldpc_info(H, info)
    
    print(f"\nFirst 5 codewords:")
    for i, cw in enumerate(codewords[:5]):
        print(f"  {i}: {cw}")
    
    print(f"\nFirst 3 local constraints:")
    for i, constraint in enumerate(constraints[:3]):
        print(f"  {i}: variables {constraint['variables']}")
        print(f"     patterns: {len(constraint['valid_patterns'])} valid patterns")