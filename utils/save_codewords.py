# This is just for saving the codes - I wouldn't worry too much about 
# the stuff in this, it just generates codewords we can test on

import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your code generators
try:
    from ldpc_codeword_generator import generate_ldpc_code, generate_regular_ldpc, print_ldpc_info
except ImportError:
    print("Warning: Could not import LDPC generator")

def save_ldpc_code(n, k, max_check_degree=3, max_var_degree=2, filename=None, regular=False):
    """Generate and save LDPC code to file"""
    if filename is None:
        if regular:
            filename = f"codes/ldpc_regular_{n}_{k}_{max_check_degree}_{max_var_degree}.pkl"
        else:
            filename = f"codes/ldpc_{n}_{k}_{max_check_degree}_{max_var_degree}.pkl"
    
    print(f"Generating LDPC code with n={n}, k={k}...")
    
    if regular:
        # For regular LDPC: max_check_degree=j (var degree), max_var_degree=k (check degree)
        codewords, local_constraints, H, code_info = generate_regular_ldpc(n, max_check_degree, max_var_degree)
    else:
        codewords, local_constraints, H, code_info = generate_ldpc_code(n, k, max_check_degree, max_var_degree)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Package everything together
    code_data = {
        'codewords': codewords,
        'local_constraints': local_constraints,
        'parity_check_matrix': H.tolist(),  # Convert to list for JSON compatibility
        'code_info': code_info,
        'code_type': 'ldpc'
    }
    
    # Save as pickle file for faster loading
    with open(filename, 'wb') as f:
        pickle.dump(code_data, f)
    
    print(f"Saved {len(codewords)} codewords to {filename}")
    print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    # Print code information
    print_ldpc_info(H, code_info)
    
    return filename

def save_repetition_code(n, filename=None):
    """Generate and save repetition code to file"""
    if filename is None:
        filename = f"codes/repetition_{n}.pkl"
    
    print(f"Generating repetition code of length {n}...")
    
    # Simple repetition code
    codewords = [
        [0] * n,  # All zeros
        [1] * n   # All ones
    ]
    
    # Local constraints: adjacent bits must be equal
    local_constraints = []
    for i in range(n-1):
        local_constraints.append({
            'variables': [i, i+1],
            'valid_patterns': [[0, 0], [1, 1]],
            'constraint_type': 'equality'
        })
    
    code_info = {
        'n': n,
        'k': 1,
        'm': n-1,
        'code_type': 'repetition',
        'num_codewords': len(codewords),
        'num_constraints': len(local_constraints)
    }
    
    code_data = {
        'codewords': codewords,
        'local_constraints': local_constraints,
        'parity_check_matrix': None,
        'code_info': code_info,
        'code_type': 'repetition'
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save as pickle file for faster loading
    with open(filename, 'wb') as f:
        pickle.dump(code_data, f)
    
    print(f"Saved {len(codewords)} codewords to {filename}")
    print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    return filename

def load_code(filename):
    """Load code data from file"""
    print(f"Loading code from {filename}...")
    
    with open(filename, 'rb') as f:
        code_data = pickle.load(f)
    
    # code_type is at the top level, not inside code_info
    code_type = code_data.get('code_type', 'unknown')
    print(f"Loaded {code_type} code:")
    print(f"  n={code_data['code_info']['n']}, k={code_data['code_info']['k']}")
    print(f"  {code_data['code_info']['num_codewords']} codewords")
    print(f"  {code_data['code_info']['num_constraints']} local constraints")
    
    return code_data

def generate_simple_hamming(r):
    """Simple Hamming code generator as fallback"""
    import itertools
    
    n = 2**r - 1
    k = n - r
    
    # Create parity check matrix
    H = np.zeros((r, n), dtype=int)
    for i in range(1, n+1):
        binary_repr = format(i, f'0{r}b')
        for j in range(r):
            H[j, i-1] = int(binary_repr[j])
    
    # Generate all codewords
    codewords = []
    for info_bits in itertools.product([0, 1], repeat=k):
        x = np.zeros(n, dtype=int)
        x[:k] = info_bits
        
        # Calculate parity bits
        for i in range(r):
            parity = 0
            for j in range(k):
                parity ^= (H[i, j] * x[j])
            x[k + i] = parity
        
        codewords.append(x.tolist())
    
    # Create local constraints
    local_constraints = []
    for i in range(r):
        variables = [j for j in range(n) if H[i, j] == 1]
        valid_patterns = []
        for pattern in itertools.product([0, 1], repeat=len(variables)):
            if sum(pattern) % 2 == 0:
                valid_patterns.append(list(pattern))
        
        local_constraints.append({
            'variables': variables,
            'valid_patterns': valid_patterns,
            'constraint_type': 'parity_check'
        })
    
    return codewords, local_constraints, H

def list_saved_codes(directory="codes"):
    """List all saved codes in directory"""
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return
    
    print(f"Saved codes in {directory}/:")
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            filepath = os.path.join(directory, filename)
            size_mb = os.path.getsize(filepath) / (1024*1024)
            print(f"  {filename} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    # Create some example codes
    print("Creating example codes...")
    
    # Small LDPC for testing
    save_ldpc_code(n=8, k=4, max_check_degree=3, max_var_degree=2)
    
    # Medium LDPC
    save_ldpc_code(n=15, k=11, max_check_degree=3, max_var_degree=2)

    # Large LDPC
    save_ldpc_code(n=32, k=24, max_check_degree=4, max_var_degree=3)

    save_ldpc_code(n=48, k=32, max_check_degree=4, max_var_degree=3)  # Rate 0.67

    save_ldpc_code(n=64, k=48, max_check_degree=5, max_var_degree=3)  # Rate 0.75

    save_ldpc_code(n=128, k=96, max_check_degree=6, max_var_degree=4) # Rate 0.75

    # WiFi 802.11n LDPC-like parameters
    #save_ldpc_code(n=648, k=432, max_check_degree=7, max_var_degree=3)  # Rate 2/3

    # DVB-S2 LDPC-like parameters  
    #save_ldpc_code(n=64800, k=43200, max_check_degree=13, max_var_degree=3) # Rate 2/3

    
    # Skip regular LDPC for now due to complexity
    print("Skipping regular LDPC generation...")
    
    # Repetition codes
    save_repetition_code(n=5)
    save_repetition_code(n=7)
    
    # List all created codes
    print("\n" + "="*50)
    list_saved_codes()
    
    # Test loading
    print("\n" + "="*50)
    print("Testing code loading...")
    try:
        code_data = load_code("codes/ldpc_8_4_3_2.pkl")
        print("Successfully loaded LDPC code!")
    except FileNotFoundError:
        print("Could not find saved code file")