from src.naive_solver.naive import findMinimumDistance
from src.lp_solver.linear_programming import (
    lp_decode,
    lp_decode_box_relaxation,
    lp_decode_simple_parity_relaxation, 
    lp_decode_subset_relaxation,
    lp_decode_syndrome_ml_relaxation,
    lp_decode_syndrome_polytope_relaxation,
    extract_parity_check_matrix
)

from utils.corrupt import corrupt_message
from utils.print_utils import (
    run_decoder_test,
    print_test_setup,
    print_comparison_summary
)
import pickle
import numpy as np

def main():
    test_ldpc_code()

def load_code_data(filename):
    """Load code data from pickle file including local constraints."""
    with open(filename, 'rb') as f:
        code_data = pickle.load(f)
    
    codewords = code_data['codewords']
    local_constraints = code_data['local_constraints'] 
    code_info = code_data['code_info']
    
    code_type = code_data.get('code_type', 'unknown')
    print(f"Loaded {code_type} code:")
    print(f"  n={code_info['n']}, k={code_info['k']}")
    print(f"  {len(codewords)} codewords, {len(local_constraints)} local constraints")
    
    return codewords, local_constraints, code_info

def test_ldpc_code():
    filename = "./codes/hamming_15_11_4_subset_2048.pkl"
    
    try:
        # Loading codewords and local constraints
        codewords, local_constraints, code_info = load_code_data(filename)
        codewords_set = set(tuple(cw) for cw in codewords)
        print("Done loading codewords, now creating corrupted message...\n")

        parity_check_matrix = extract_parity_check_matrix(codewords)
        print(f"Generated parity check matrix H: {parity_check_matrix.shape}")

        # Choose a random original message
        np.random.seed(42)  # For reproducibility
        original_idx = np.random.randint(len(codewords))
        original = codewords[original_idx]
        corrupted = corrupt_message(original, 1) 
        
        # Ensure that the corrupted word isn't a valid codeword
        while tuple(corrupted) in codewords_set:
            corrupted = corrupt_message(original, 3)

        # Print test setup
        print_test_setup(original, corrupted, codewords, code_info)
        
        # Store results for comparison
        results = []
        
        # Test 1: Naive decoder
        decoded, cost, exec_time = run_decoder_test(
            decoder_func=findMinimumDistance,
            method_name="NAIVE DECODER",
            corrupted_word=corrupted,
            original_word=original,
            codewords=codewords
        )
        results.append(("Naive", decoded, cost, exec_time, decoded == original))
        
        # Test 2: LP decoder (fundamental relaxation)
        decoded, cost, exec_time = run_decoder_test(
            decoder_func=lp_decode,
            method_name="LP DECODER (fundamental relaxation)",
            corrupted_word=corrupted,
            original_word=original,
            codewords=codewords,
            channel_error_prob=0.1,
            relaxation='fundamental',
            local_constraints=local_constraints
        )
        results.append(("LP Fundamental", decoded, cost, exec_time, decoded == original))
        
        # Test 3: LP decoder (box relaxation)
        decoded, cost, exec_time = run_decoder_test(
            decoder_func=lp_decode_box_relaxation,
            method_name="LP DECODER (box relaxation)",
            corrupted_word=corrupted,
            original_word=original,
            codewords=codewords,
            channel_error_prob=0.1
        )
        results.append(("LP Box", decoded, cost, exec_time, decoded == original))

        # Test 4: LP decoder (simple parity relaxation)
        decoded, cost, exec_time = run_decoder_test(
            decoder_func=lp_decode_simple_parity_relaxation,
            method_name="LP DECODER (simple parity relaxation)",
            corrupted_word=corrupted,
            original_word=original,
            codewords=codewords,
            channel_error_prob=0.1,
            local_constraints=local_constraints
        )
        results.append(("LP Simple Parity", decoded, cost, exec_time, decoded == original))

        # Test 5: LP decoder (syndrome ML relaxation)
        decoded, cost, exec_time = run_decoder_test(
            decoder_func=lp_decode_syndrome_ml_relaxation,
            method_name="LP DECODER (syndrome ML relaxation)",
            corrupted_word=corrupted,
            original_word=original,
            codewords=codewords,
            channel_error_prob=0.1,
            parity_check_matrix=parity_check_matrix,
            max_error_weight=3
        )
        results.append(("LP Syndrome ML", decoded, cost, exec_time, decoded == original))

        # Test 6: LP decoder (syndrome polytope relaxation)
        decoded, cost, exec_time = run_decoder_test(
            decoder_func=lp_decode_syndrome_polytope_relaxation,
            method_name="LP DECODER (syndrome polytope relaxation)",
            corrupted_word=corrupted,
            original_word=original,
            codewords=codewords,
            channel_error_prob=0.1,
            parity_check_matrix=parity_check_matrix
        )
        results.append(("LP Syndrome Polytope", decoded, cost, exec_time, decoded == original))

        # Print comparison summary
        print_comparison_summary(results)
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        print("Run the code generator first:")
        print("python utils/save_codewords.py")
        print("This will generate LDPC codes in the implementation/codes/ directory")
    except Exception as e:
        print(f"Error in LDPC code test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":  
    main()