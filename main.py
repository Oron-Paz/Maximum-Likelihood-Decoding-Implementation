from src.naive_solver.naive import findMinimumDistance
from src.lp_solver.linear_programming import lp_decode
from utils.corrupt import corrupt_message
import pickle
import time
import numpy as np

def load_ldpc_code(filename):
    """Load LDPC code from pickle file."""
    try:
        with open(filename, 'rb') as f:
            code_data = pickle.load(f)
        
        # Extract data based on structure
        if isinstance(code_data, dict):
            codewords = code_data.get('codewords', [])
            H = np.array(code_data.get('H', []))
            G = np.array(code_data.get('G', []))
            n = code_data.get('n', len(codewords[0]) if codewords else 0)
            k = code_data.get('k', 0)
            description = code_data.get('description', 'LDPC Code')
        else:
            # Assume it's just the codewords array
            codewords = code_data.tolist() if hasattr(code_data, 'tolist') else code_data
            H = None
            G = None
            n = len(codewords[0]) if codewords else 0
            k = 0
            description = 'LDPC Code'
        
        print(f"Loaded: {description}")
        print(f"Parameters: n={n}, k={k}")
        print(f"Number of codewords: {len(codewords)}")
        
        return codewords, H, G, n, k, description
    
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, None, None, 0, 0, ""

def generate_parity_check_matrix(codewords_list, n, k):
    """Generate a parity check matrix if not provided."""
    print("Generating parity check matrix from codewords...")
    
    # Convert codewords to numpy array
    C = np.array(codewords_list, dtype=int)
    
    # Find a basis for the null space (this gives us H such that HC^T = 0)
    # This is a simplified approach - real LDPC codes have specific structure
    
    # For small codes, we can use Gaussian elimination
    if len(codewords_list) <= 1000:
        try:
            # Use SVD to find null space
            from scipy.linalg import null_space
            H_approx = null_space(C.astype(float).T)
            H = (H_approx > 0.5).astype(int)
            
            # Make sure H has the right dimensions
            if H.shape[1] != n:
                H = H.T
            
            # Verify it works (at least approximately)
            syndrome = (H @ C.T) % 2
            if np.sum(syndrome) == 0:
                print(f"Generated H matrix: {H.shape}")
                return H
        except:
            pass
    
    # Fallback: create a simple parity check matrix
    print("Using fallback parity check matrix generation...")
    m = n - k if k > 0 else n // 2
    H = np.random.randint(0, 2, size=(m, n))
    
    # Ensure no all-zero rows
    for i in range(m):
        if np.sum(H[i, :]) == 0:
            H[i, np.random.randint(0, n)] = 1
    
    print(f"Generated fallback H matrix: {H.shape}")
    return H

def main():
    print("Testing Fundamental Polytope LP Approach vs Naive Method")
    print("\n" + "="*60)
    test_ldpc_code()

def test_ldpc_code():
    print("TEST: LDPC Code with Fundamental Polytope Relaxation")
    print("-" * 50)
    
    # Load the LDPC code
    filename = "codes/ldpc_24_12_6_3.pkl"
    
    try:
        codewords_list, H, G, n, k, description = load_ldpc_code(filename)
        
        if codewords_list is None:
            print(f"Failed to load {filename}")
            return
        
        # Generate parity check matrix if not available
        if H is None or H.size == 0:
            H = generate_parity_check_matrix(codewords_list, n, k)
        
        # Create set for fast lookup
        codewords_set = set(tuple(cw) for cw in codewords_list)
        print("Done loading codewords, now creating corrupted message...\n")

        # Select a random original codeword
        original_idx = np.random.randint(len(codewords_list))
        original = codewords_list[original_idx]
        
        # Create corrupted version with 1-2 errors
        num_errors = 2
        corrupted = corrupt_message(original, num_errors)
        
        # Ensure corrupted word isn't a valid codeword
        while tuple(corrupted) in codewords_set:
            corrupted = corrupt_message(original, num_errors)

        print(f"Original:  {original}")
        print(f"Corrupted: {corrupted}")
        print(f"Errors introduced: {num_errors}")
        print(f"Codebook size: {len(codewords_list):,} codewords")
        print(f"Parity check matrix: {H.shape}")
        
        # Test 1: Naive decoder
        print("\n" + "="*60)
        print("NAIVE DECODER:")
        print("-" * 45)
        start = time.perf_counter()
        naive_decoded_word, hamming_distance = findMinimumDistance(corrupted, codewords_list)
        end = time.perf_counter()
        naive_execution_time = end - start

        print(f"Decoded word: {naive_decoded_word}")
        print(f"Hamming Distance: {hamming_distance}")
        print(f"Execution time: {naive_execution_time:.3f}s")
        print(f"Correct decoding: {naive_decoded_word == original}")
        
        # Test 2: LP decoder with fundamental polytope relaxation
        print("\n" + "="*60)
        print("LP DECODER (Fundamental Polytope Relaxation):")
        print("-" * 45)
        
        start = time.perf_counter()
        lp_decoded_word, lp_cost = lp_decode(
            corrupted, 
            codewords_list, 
            channel_error_prob=0.1, 
            relaxation='fundamental',
            parity_check_matrix=H
        )
        end = time.perf_counter()
        lp_execution_time = end - start
        
        if lp_decoded_word is not None:
            print(f"Decoded word: {lp_decoded_word}")
            print(f"LP cost: {lp_cost:.6f}")
            print(f"Execution time: {lp_execution_time:.3f}s")
            print(f"Correct decoding: {lp_decoded_word == original}")
        else:
            print("LP decoding failed!")
            lp_decoded_word = []
            lp_execution_time = float('inf')
        
        # Test 3: LP decoder with exact polytope (for comparison, if feasible)
        print("\n" + "="*60)
        print("LP DECODER (Exact Polytope - for comparison):")
        print("-" * 45)
        
        if len(codewords_list) <= 500:  # Only for small codes
            start = time.perf_counter()
            exact_lp_decoded_word, exact_lp_cost = lp_decode(
                corrupted, 
                codewords_list, 
                channel_error_prob=0.1, 
                relaxation='exact'
            )
            end = time.perf_counter()
            exact_lp_execution_time = end - start
            
            if exact_lp_decoded_word is not None:
                print(f"Decoded word: {exact_lp_decoded_word}")
                print(f"LP cost: {exact_lp_cost:.6f}")
                print(f"Execution time: {exact_lp_execution_time:.3f}s")
                print(f"Correct decoding: {exact_lp_decoded_word == original}")
            else:
                print("Exact LP decoding failed!")
                exact_lp_decoded_word = []
        else:
            print("Skipping exact LP (too many codewords)")
            exact_lp_decoded_word = None
            exact_lp_execution_time = float('inf')
        
        # Comparison
        print("\n" + "="*60)
        print("COMPARISON:")
        print("-" * 45)
        print(f"Naive vs Fundamental LP agree: {naive_decoded_word == lp_decoded_word}")
        if exact_lp_decoded_word is not None:
            print(f"Fundamental LP vs Exact LP agree: {lp_decoded_word == exact_lp_decoded_word}")
        
        print(f"\nTiming comparison:")
        print(f"Naive time:        {naive_execution_time:.3f}s")
        print(f"Fundamental LP:    {lp_execution_time:.3f}s")
        if exact_lp_decoded_word is not None:
            print(f"Exact LP:          {exact_lp_execution_time:.3f}s")
        
        print(f"\nSpeedup analysis:")
        if lp_execution_time > 0:
            speedup = naive_execution_time / lp_execution_time
            print(f"Fundamental LP vs Naive: {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
        
        if exact_lp_decoded_word is not None and exact_lp_execution_time > 0:
            relaxation_speedup = exact_lp_execution_time / lp_execution_time
            print(f"Fundamental LP vs Exact LP: {relaxation_speedup:.2f}x {'faster' if relaxation_speedup > 1 else 'slower'}")
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        print("Available files in codes directory:")
        import os
        if os.path.exists("./codes/"):
            for file in os.listdir("./codes/"):
                if file.endswith('.pkl'):
                    print(f"  - {file}")
        else:
            print("  codes/ directory not found")
    except Exception as e:
        print(f"Error in LDPC code test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":  
    main()