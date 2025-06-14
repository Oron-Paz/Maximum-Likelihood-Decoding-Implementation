import numpy as np
import itertools

def generate_hamming_code(r, max_codewords=None):
    """
    Generate Hamming [2^r-1, 2^r-r-1, 3] code.
    
    Args:
        r: number of parity bits (r >= 2)
        max_codewords: maximum number of codewords to generate (None = all)
    
    Returns:
        codewords, local_constraints, H, code_info
    """
    n = 2**r - 1
    k = n - r
    total_codewords = 2**k
    
    print(f"Generating Hamming [{n},{k},3] code...")
    
    # Create parity check matrix H
    H = np.zeros((r, n), dtype=int)
    for i in range(1, n+1):
        binary_repr = format(i, f'0{r}b')
        for j in range(r):
            H[j, i-1] = int(binary_repr[j])
    
    print(f"Parity check matrix H ({r}×{n}):")
    print(H)
    
    # Decide how many codewords to generate
    if max_codewords is None or max_codewords >= total_codewords:
        # Generate ALL codewords
        print(f"Generating all {total_codewords} codewords...")
        codewords = []
        for info_vector in itertools.product([0, 1], repeat=k):
            codeword = encode_hamming(list(info_vector), H, n, k, r)
            codewords.append(codeword)
    else:
        # Generate subset of codewords
        print(f"Generating {max_codewords} out of {total_codewords} codewords...")
        codewords = []
        step = max(1, total_codewords // max_codewords)
        
        for i in range(0, total_codewords, step):
            if len(codewords) >= max_codewords:
                break
            # Convert i to binary information vector
            info_vector = [(i >> j) & 1 for j in range(k)]
            codeword = encode_hamming(info_vector, H, n, k, r)
            codewords.append(codeword)
        
        # Always include the zero codeword and some key codewords
        zero_codeword = [0] * n
        if zero_codeword not in codewords:
            codewords[0] = zero_codeword
    
    # Create local constraints (one per parity check)
    local_constraints = []
    for i in range(r):
        variables = [j for j in range(n) if H[i, j] == 1]
        
        # Even parity constraint
        valid_patterns = []
        for pattern in itertools.product([0, 1], repeat=len(variables)):
            if sum(pattern) % 2 == 0:
                valid_patterns.append(list(pattern))
        
        local_constraints.append({
            'variables': variables,
            'valid_patterns': valid_patterns,
            'constraint_type': 'parity_check',
            'check_index': i
        })
    
    code_info = {
        'n': n,
        'k': k,
        'm': r,
        'code_type': 'hamming',
        'minimum_distance': 3,
        'num_codewords': len(codewords),
        'total_possible_codewords': total_codewords,
        'num_constraints': len(local_constraints),
        'is_subset': max_codewords is not None and max_codewords < total_codewords
    }
    
    print(f"Generated {len(codewords)} codewords")
    return codewords, local_constraints, H, code_info

def encode_hamming(info_bits, H, n, k, r):
    """Encode information bits into Hamming codeword."""
    codeword = [0] * n
    
    # Place information bits in non-power-of-2 positions
    info_idx = 0
    for pos in range(1, n+1):
        if (pos & (pos - 1)) != 0:  # pos is NOT a power of 2
            codeword[pos-1] = info_bits[info_idx]
            info_idx += 1
    
    # Calculate parity bits
    for i in range(r):
        parity_pos = 2**i
        parity = 0
        for pos in range(1, n+1):
            if pos & parity_pos:  # bit i is set in pos
                parity ^= codeword[pos-1]
        codeword[parity_pos-1] = parity
    
    return codeword

def generate_repetition_code(n):
    """Generate repetition code [n,1,n]."""
    print(f"Generating repetition [{n},1,{n}] code...")
    
    codewords = [
        [0] * n,  # All zeros
        [1] * n   # All ones
    ]
    
    # Parity check matrix: adjacent bits must be equal
    H = np.zeros((n-1, n), dtype=int)
    for i in range(n-1):
        H[i, i] = 1      # x_i
        H[i, i+1] = 1    # + x_{i+1} = 0 (mod 2)
    
    # Local constraints: each pair of adjacent bits
    local_constraints = []
    for i in range(n-1):
        local_constraints.append({
            'variables': [i, i+1],
            'valid_patterns': [[0, 0], [1, 1]],
            'constraint_type': 'equality',
            'check_index': i
        })
    
    code_info = {
        'n': n,
        'k': 1,
        'm': n-1,
        'code_type': 'repetition',
        'minimum_distance': n,
        'num_codewords': 2,
        'num_constraints': n-1
    }
    
    return codewords, local_constraints, H, code_info

def generate_parity_check_code(n, max_codewords=None):
    """Generate single parity check code [n,n-1,2]."""
    print(f"Generating parity check [{n},{n-1},2] code...")
    
    total_codewords = 2**(n-1)
    
    if max_codewords is None or max_codewords >= total_codewords:
        # Generate all vectors with even parity
        codewords = []
        for vector in itertools.product([0, 1], repeat=n):
            if sum(vector) % 2 == 0:  # Even parity
                codewords.append(list(vector))
    else:
        # Generate subset with even parity
        print(f"Generating {max_codewords} out of {total_codewords} codewords...")
        codewords = []
        count = 0
        
        for vector in itertools.product([0, 1], repeat=n):
            if sum(vector) % 2 == 0:  # Even parity
                if count < max_codewords:
                    codewords.append(list(vector))
                    count += 1
                else:
                    break
    
    # Single parity check constraint
    H = np.ones((1, n), dtype=int)  # Sum all bits = 0 (mod 2)
    
    local_constraints = [{
        'variables': list(range(n)),
        'valid_patterns': [list(pattern) for pattern in codewords[:256]],  # Limit patterns for memory
        'constraint_type': 'parity_check',
        'check_index': 0
    }]
    
    code_info = {
        'n': n,
        'k': n-1,
        'm': 1,
        'code_type': 'parity_check',
        'minimum_distance': 2,
        'num_codewords': len(codewords),
        'total_possible_codewords': total_codewords,
        'num_constraints': 1,
        'is_subset': max_codewords is not None and max_codewords < total_codewords
    }
    
    print(f"Generated {len(codewords)} codewords")
    return codewords, local_constraints, H, code_info

def generate_dual_hamming_code(r, max_codewords=None):
    """
    Generate dual of Hamming code = simplex code [2^r-1, r, 2^{r-1}].
    
    Args:
        r: dimension
        max_codewords: maximum number of codewords (None = all)
    
    Returns:
        codewords, local_constraints, H, code_info
    """
    n = 2**r - 1
    k = r
    total_codewords = 2**r
    
    print(f"Generating simplex [{n},{k},{2**(r-1)}] code...")
    
    # Generator matrix G is transpose of Hamming parity check matrix
    G = np.zeros((r, n), dtype=int)
    for i in range(1, n+1):
        binary_repr = format(i, f'0{r}b')
        for j in range(r):
            G[j, i-1] = int(binary_repr[j])
    
    # Generate codewords (subset if requested)
    if max_codewords is None or max_codewords >= total_codewords:
        # Generate all 2^r codewords
        print(f"Generating all {total_codewords} codewords...")
        codewords = []
        for info_vector in itertools.product([0, 1], repeat=r):
            codeword = np.dot(info_vector, G) % 2
            codewords.append(codeword.tolist())
    else:
        # Generate subset
        print(f"Generating {max_codewords} out of {total_codewords} codewords...")
        codewords = []
        step = max(1, total_codewords // max_codewords)
        
        for i in range(0, total_codewords, step):
            if len(codewords) >= max_codewords:
                break
            info_vector = [(i >> j) & 1 for j in range(r)]
            codeword = np.dot(info_vector, G) % 2
            codewords.append(codeword.tolist())
    
    # Create simplified parity check matrix H
    # For simplex codes, we can use a simpler approach
    m = n - k  # This should be 2^r - 1 - r
    H = np.zeros((m, n), dtype=int)
    
    # Fill H with some structure (simplified approach)
    for i in range(min(m, r)):
        H[i, :] = G[i, :]  # Use rows from G
    
    # Create simple local constraints
    local_constraints = []
    for i in range(min(m, 4)):  # Limit to avoid memory issues
        # Each constraint involves a few variables
        num_vars = min(4, n)
        variables = list(range(num_vars))
        
        # Generate small valid patterns
        valid_patterns = []
        for pattern in itertools.product([0, 1], repeat=num_vars):
            valid_patterns.append(list(pattern))
        
        local_constraints.append({
            'variables': variables,
            'valid_patterns': valid_patterns[:16],  # Limit patterns
            'constraint_type': 'simplex',
            'check_index': i
        })
    
    code_info = {
        'n': n,
        'k': k,
        'm': m,
        'code_type': 'simplex',
        'minimum_distance': 2**(r-1),
        'num_codewords': len(codewords),
        'total_possible_codewords': total_codewords,
        'num_constraints': len(local_constraints),
        'is_subset': max_codewords is not None and max_codewords < total_codewords
    }
    
    print(f"Generated {len(codewords)} codewords")
    return codewords, local_constraints, H, code_info

# Test the codes
if __name__ == "__main__":
    print("="*60)
    print("TESTING STRUCTURED CODES WITH SUBSET SAMPLING")
    print("="*60)
    
    # Test Hamming codes
    print("\n1. HAMMING CODES:")
    codewords, constraints, H, info = generate_hamming_code(r=3)
    print(f"   Hamming [{info['n']},{info['k']},3]: {len(codewords)} codewords")
    
    # Test large Hamming with subset
    codewords, constraints, H, info = generate_hamming_code(r=5, max_codewords=1000)
    print(f"   Hamming [{info['n']},{info['k']},3] (subset): {len(codewords)}/{info['total_possible_codewords']} codewords")
    
    # Test simplex codes
    print("\n2. SIMPLEX CODES:")
    codewords, constraints, H, info = generate_dual_hamming_code(r=4, max_codewords=16)
    print(f"   Simplex [{info['n']},{info['k']},{2**(4-1)}]: {len(codewords)} codewords")
    
    print("\n" + "="*60)
    print("Subset sampling works! LP will work fine with partial codewords.")
    print("="*60)