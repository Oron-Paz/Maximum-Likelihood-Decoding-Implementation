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
    
    # For very small codes, ensure we have enough constraints
    if m < 3:
        print(f"Warning: Only {m} parity checks for n={n}, k={k}. Adding more constraints.")
        m = max(3, n // 3)  # Ensure at least 3 checks
    
    # Create sparse parity check matrix H
    H = np.zeros((m, n), dtype=int)
    
    # Fill H with sparse pattern - ensure each check has at least 2 variables
    for i in range(m):
        # Each check involves 2 to max_check_degree variables
        num_vars = np.random.randint(2, min(max_check_degree + 1, n + 1))
        var_indices = np.random.choice(n, size=num_vars, replace=False)
        H[i, var_indices] = 1
    
    # Ensure each variable participates in at least 1 check and at most max_var_degree checks
    for j in range(n):
        current_degree = np.sum(H[:, j])
        
        if current_degree == 0:
            # Variable not connected, connect to a random check
            check_idx = np.random.randint(m)
            H[check_idx, j] = 1
        elif current_degree > max_var_degree:
            # Remove excess connections
            check_indices = np.where(H[:, j] == 1)[0]
            excess = current_degree - max_var_degree
            remove_indices = np.random.choice(check_indices, size=excess, replace=False)
            H[remove_indices, j] = 0
    
    density = np.mean(H)
    print(f"Parity check matrix density: {density:.3f}")
    
    # Generate codewords using systematic encoding
    codewords = []
    
    # For systematic LDPC, we need to solve Hx = 0
    # This is complex for general LDPC, so we'll use a simplified approach
    
    if k <= 12:  # For small codes, try all information sequences
        for info_bits in itertools.product([0, 1], repeat=k):
            x = solve_ldpc_encoding(info_bits, H, n, k, m)
            if x is not None:
                codewords.append(x.tolist())
    else:
        # For larger codes, generate a subset
        print(f"Warning: k={k} too large for full enumeration, generating sample")
        num_samples = min(1000, 2**min(k, 10))
        attempts = 0
        while len(codewords) < num_samples and attempts < num_samples * 3:
            info_bits = np.random.randint(0, 2, k)
            x = solve_ldpc_encoding(info_bits, H, n, k, m)
            if x is not None:
                codeword_tuple = tuple(x.tolist())
                if codeword_tuple not in {tuple(cw) for cw in codewords}:
                    codewords.append(x.tolist())
            attempts += 1
    
    # If we couldn't generate enough codewords, fall back to simpler method
    if len(codewords) < 4:
        print("Warning: Systematic encoding failed, using brute force approach")
        codewords = generate_codewords_brute_force(H, n, min(16, 2**min(k, 4)))
    
    # Create local constraints from parity checks
    local_constraints = create_ldpc_local_constraints(H)
    
    code_info = {
        'n': n,
        'k': k,
        'm': m,
        'density': density,
        'max_check_degree': max_check_degree,
        'max_var_degree': max_var_degree,
        'num_codewords': len(codewords),
        'num_constraints': len(local_constraints)
    }
    
    print(f"Generated {len(codewords)} unique codewords")
    print(f"Created {len(local_constraints)} local constraints")
    
    return codewords, local_constraints, H, code_info

def solve_ldpc_encoding(info_bits, H, n, k, m):
    """
    Solve for a valid codeword given information bits.
    This is a simplified systematic encoder.
    """
    x = np.zeros(n, dtype=int)
    x[:k] = info_bits
    
    # Try to solve for remaining bits
    for i in range(m):
        syndrome = 0
        parity_positions = []
        
        # Calculate syndrome from known bits
        for j in range(n):
            if j < k:
                syndrome ^= (H[i, j] * x[j])
            elif H[i, j] == 1:
                parity_positions.append(j)
        
        # Set one parity bit to satisfy the constraint
        if parity_positions:
            x[parity_positions[0]] = syndrome
    
    # Verify this is actually a valid codeword
    syndrome_check = np.dot(H, x) % 2
    if np.all(syndrome_check == 0):
        return x
    else:
        return None

def generate_codewords_brute_force(H, n, max_codewords=16):
    """Brute force search for valid codewords."""
    codewords = []
    
    # Try up to 2^min(n,16) possibilities
    max_attempts = min(2**n, 2**16, max_codewords * 100)
    
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
    x = np.zeros(n, dtype=int)
    x[:k] = info_bits
    
    m = n - k
    
    # Try to solve for parity bits
    # This is a simplified approach - in practice, you'd need more sophisticated encoding
    for i in range(m):
        parity = 0
        # Calculate parity from information bits
        for j in range(k):
            parity ^= (H[i, j] * x[j])
        
        # Find parity bit positions for this check
        parity_positions = [j for j in range(k, n) if H[i, j] == 1]
        
        if parity_positions:
            # Set the first available parity bit
            x[parity_positions[0]] = parity
    
    # Verify this is a valid codeword
    syndrome = np.dot(H, x) % 2
    if np.all(syndrome == 0):
        return x
    else:
        # Try a different approach or return None
        return None

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
        
        if len(variables) == 0:
            continue
        
        # Generate all valid patterns for this parity check (even parity)
        valid_patterns = []
        for pattern in itertools.product([0, 1], repeat=len(variables)):
            if sum(pattern) % 2 == 0:  # Even parity constraint
                valid_patterns.append(list(pattern))
        
        if valid_patterns:
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
    
    # Generate codewords and constraints
    info_bits = n - m
    codewords = []
    
    # Generate sample codewords
    for _ in range(min(100, 2**info_bits)):
        info = np.random.randint(0, 2, info_bits)
        x = encode_systematic_ldpc(info, H, n, info_bits)
        if x is not None:
            codewords.append(x.tolist())
    
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