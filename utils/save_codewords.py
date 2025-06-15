# Save codewords file, currently needs to be ran before running main file, 
# adjust the function calls at the bottom to generate whichever files
# you want to test on. But for the sake of the assigment doesnt really matter

import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your Hamming code generator
try:
    from hamming_codeword_generator import generate_hamming_code, generate_repetition_code, generate_parity_check_code, generate_dual_hamming_code
except ImportError:
    print("Warning: Could not import Hamming generators")

def save_hamming_code_subset(r, max_codewords, filename=None):
    """Generate and save subset of Hamming code to file"""
    n = 2**r - 1
    k = n - r
    
    if filename is None:
        filename = f"codes/hamming_{n}_{k}_{r}_subset_{max_codewords}.pkl"
    
    print(f"Generating Hamming [{n},{k},3] code subset with r={r}, max_codewords={max_codewords}...")
    
    codewords, local_constraints, H, code_info = generate_hamming_code(r, max_codewords=max_codewords)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Package everything together
    code_data = {
        'codewords': codewords,
        'local_constraints': local_constraints,
        'parity_check_matrix': H.tolist() if H is not None else None,
        'code_info': code_info,
        'code_type': 'hamming'
    }
    
    # Save as pickle file for faster loading
    with open(filename, 'wb') as f:
        pickle.dump(code_data, f)
    
    print(f"Saved {len(codewords)} codewords to {filename}")
    print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    # Print code information
    print_code_info(code_info)
    
    return filename

def save_repetition_code(n, filename=None):
    """Generate and save repetition code to file"""
    if filename is None:
        filename = f"codes/repetition_{n}.pkl"
    
    print(f"Generating repetition code of length {n}...")
    
    codewords, local_constraints, H, code_info = generate_repetition_code(n)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    code_data = {
        'codewords': codewords,
        'local_constraints': local_constraints,
        'parity_check_matrix': H.tolist() if H is not None else None,
        'code_info': code_info,
        'code_type': 'repetition'
    }
    
    # Save as pickle file for faster loading
    with open(filename, 'wb') as f:
        pickle.dump(code_data, f)
    
    print(f"Saved {len(codewords)} codewords to {filename}")
    print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    print_code_info(code_info)
    
    return filename

def save_parity_check_code(n, filename=None):
    """Generate and save parity check code to file"""
    if filename is None:
        filename = f"codes/parity_check_{n}_{n-1}.pkl"
    
    print(f"Generating parity check [{n},{n-1},2] code...")
    
    codewords, local_constraints, H, code_info = generate_parity_check_code(n)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    code_data = {
        'codewords': codewords,
        'local_constraints': local_constraints,
        'parity_check_matrix': H.tolist() if H is not None else None,
        'code_info': code_info,
        'code_type': 'parity_check'
    }
    
    # Save as pickle file for faster loading
    with open(filename, 'wb') as f:
        pickle.dump(code_data, f)
    
    print(f"Saved {len(codewords)} codewords to {filename}")
    print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    print_code_info(code_info)
    
    return filename

def save_simplex_code(r, filename=None):
    """Generate and save simplex code to file"""
    n = 2**r - 1
    k = r
    
    if filename is None:
        filename = f"codes/simplex_{n}_{k}_{r}.pkl"
    
    print(f"Generating simplex [{n},{k},{2**(r-1)}] code with r={r}...")
    
    codewords, local_constraints, H, code_info = generate_dual_hamming_code(r)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    code_data = {
        'codewords': codewords,
        'local_constraints': local_constraints,
        'parity_check_matrix': H.tolist() if H is not None else None,
        'code_info': code_info,
        'code_type': 'simplex'
    }
    
    # Save as pickle file for faster loading
    with open(filename, 'wb') as f:
        pickle.dump(code_data, f)
    
    print(f"Saved {len(codewords)} codewords to {filename}")
    print(f"File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    print_code_info(code_info)
    
    return filename

def print_code_info(code_info):
    """Print detailed information about the code."""
    print("\n" + "="*50)
    print("CODE INFORMATION")
    print("="*50)
    
    print(f"Code type: {code_info.get('code_type', 'unknown')}")
    print(f"Block length (n): {code_info['n']}")
    print(f"Information bits (k): {code_info['k']}")
    print(f"Parity checks (m): {code_info['m']}")
    print(f"Rate: {code_info['k']/code_info['n']:.3f}")
    print(f"Minimum distance: {code_info.get('minimum_distance', 'unknown')}")
    print(f"Number of codewords: {code_info['num_codewords']}")
    print(f"Number of local constraints: {code_info['num_constraints']}")

def load_code(filename):
    """Load code data from file"""
    print(f"Loading code from {filename}...")
    
    with open(filename, 'rb') as f:
        code_data = pickle.load(f)
    
    code_type = code_data.get('code_type', 'unknown')
    print(f"Loaded {code_type} code:")
    print(f"  n={code_data['code_info']['n']}, k={code_data['code_info']['k']}")
    print(f"  {code_data['code_info']['num_codewords']} codewords")
    print(f"  {code_data['code_info']['num_constraints']} local constraints")
    
    return code_data

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
    print("Creating structured codes for LP decoding...")
    print("="*60)
    
    # Small codes - perfect for testing and development
    print("\n1. SMALL CODES (for testing):")
    save_hamming_code_subset(r=3, max_codewords=16)      # [7,4,3] - 16 codewords
    save_repetition_code(n=5)   # [5,1,5] - 2 codewords  
    save_repetition_code(n=7)   # [7,1,7] - 2 codewords
    save_parity_check_code(n=4) # [4,3,2] - 8 codewords
    save_parity_check_code(n=6) # [6,5,2] - 32 codewords
    
    # Medium codes - good balance of complexity and manageability
    print("\n2. MEDIUM CODES (for serious testing):")
    save_hamming_code_subset(r=4, max_codewords=2048)      # [15,11,3] - 2048 codewords
    save_simplex_code(r=4)      # [15,4,8] - 16 codewords
    save_parity_check_code(n=8) # [8,7,2] - 128 codewords
    save_repetition_code(n=9)   # [9,1,9] - 2 codewords
    
    # Larger codes - for performance testing
    print("\n3. LARGER CODES (for performance testing):")
    save_hamming_code_subset(r=5, max_codewords=10000000)  # [31,26,3] - subset of 67M codewords
    save_simplex_code(r=5)      # [31,5,16] - 32 codewords
    save_parity_check_code(n=10) # [10,9,2] - 512 codewords
    save_parity_check_code(n=12) # [12,11,2] - 2048 codewords
    
    # Very large codes - only if you need them
    print("\n4. LARGE CODES (only if needed - might be slow):")
    try:
        # Comment out if too slow
        # save_hamming_code(r=6)      # [63,57,3] - huge number of codewords
        save_simplex_code(r=6)      # [63,6,32] - 64 codewords
        save_repetition_code(n=15)  # [15,1,15] - 2 codewords
        print("Large codes generated successfully!")
    except Exception as e:
        print(f"Skipped large codes due to: {e}")
    
    # List all created codes
    print("\n" + "="*60)
    print("SUMMARY:")
    list_saved_codes()
    
    # Test loading one code
    print("\n" + "="*60)
    print("Testing code loading...")
    try:
        code_data = load_code("codes/hamming_7_4_3.pkl")
        print("✓ Successfully loaded Hamming code!")
        
        # Show first few codewords
        codewords = code_data['codewords']
        print(f"\nFirst 8 codewords:")
        for i, cw in enumerate(codewords[:8]):
            print(f"  {i:2d}: {cw}")
            
    except FileNotFoundError:
        print("✗ Could not find saved code file")
    except Exception as e:
        print(f"✗ Error loading code: {e}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS FOR LP DECODING:")
    print("- Start with hamming_7_4_3.pkl (small, perfect for debugging)")
    print("- Use hamming_15_11_4.pkl for serious testing") 
    print("- Try simplex codes for very clean constraints")
    print("- Parity check codes have simple single constraint")
    print("="*60)