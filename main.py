from src.naive_solver.naive import findMinimumDistance
from src.lp_solver.linear_programming import lp_decode
from utils.corrupt import corrupt_message
import pickle
import time
import numpy as np

def main():
    print("Testing Paper's LP Approach vs Naive Method")
    print("\n" + "="*60)
    test_ldpc_code()

def load_code_data(filename):
    """Load code data from pickle file including local constraints."""
    with open(filename, 'rb') as f:
        code_data = pickle.load(f)
    
    codewords = code_data['codewords']
    local_constraints = code_data['local_constraints'] 
    code_info = code_data['code_info']
    
    # code_type is at the top level, not inside code_info
    code_type = code_data.get('code_type', 'unknown')
    print(f"Loaded {code_type} code:")
    print(f"  n={code_info['n']}, k={code_info['k']}")
    print(f"  {len(codewords)} codewords, {len(local_constraints)} local constraints")
    
    return codewords, local_constraints, code_info

def test_ldpc_code():
    print("TEST: LDPC Code [15,11] with Fundamental Relaxation")
    print("-" * 45)
    
    filename = "./codes/ldpc_128_96_6_4.pkl"
    
    try:
        # Loading codewords and local constraints
        codewords, local_constraints, code_info = load_code_data(filename)
        codewords_set = set(tuple(cw) for cw in codewords)
        print("Done loading codewords, now creating corrupted message...\n")

        # Choose a random original message
        np.random.seed(42)  # For reproducibility
        original_idx = np.random.randint(len(codewords))
        original = codewords[original_idx]
        corrupted = corrupt_message(original, 1)  # Add 1 error
        
        # Ensure that the corrupted word isn't a valid codeword
        while tuple(corrupted) in codewords_set:
            corrupted = corrupt_message(original, 1)

        print(f"Original:  {original}")
        print(f"Corrupted: {corrupted}")
        print(f"Codebook size: {len(codewords):,} codewords")
        print(f"Code parameters: n={code_info['n']}, k={code_info['k']}, rate={code_info['k']/code_info['n']:.3f}")
        
        # Test 1: Naive decoder
        print("\n" + "="*60)
        print("NAIVE DECODER:")
        print("-" * 45)
        start = time.perf_counter()
        naive_decoded_word, hamming_distance = findMinimumDistance(corrupted, codewords)
        end = time.perf_counter()
        naive_execution_time = end - start

        print(f"Decoded word: {naive_decoded_word}")
        print(f"Hamming Distance: {hamming_distance}")
        print(f"Execution time: {naive_execution_time:.6f}s")
        print(f"Correct decoding: {naive_decoded_word == original}")
        
        # Test 3: LP decoder (fundamental relaxation) - THE MAIN EVENT!
        print("\n" + "="*60)
        print("LP DECODER (fundamental relaxation):")
        print("-" * 45)
        
        try:
            start = time.perf_counter()
            lp_relaxed_decoded_word, lp_relaxed_cost = lp_decode(
                corrupted, codewords, channel_error_prob=0.1, 
                relaxation='fundamental', local_constraints=local_constraints
            )
            end = time.perf_counter()
            lp_relaxed_execution_time = end - start
            
            print(f"Decoded word: {lp_relaxed_decoded_word}")
            print(f"LP cost: {lp_relaxed_cost:.6f}")
            print(f"Execution time: {lp_relaxed_execution_time:.6f}s")
            print(f"Correct decoding: {lp_relaxed_decoded_word == original}")
            
        except Exception as e:
            print(f"Fundamental relaxation failed: {e}")
            lp_relaxed_decoded_word = None
            lp_relaxed_execution_time = float('inf')
        
        # Comparison
        print("\n" + "="*60)
        print("COMPARISON:")
        print("-" * 45)
        
        if naive_decoded_word and lp_relaxed_decoded_word:
            print(f"Naive vs Relaxed LP agree: {naive_decoded_word == lp_relaxed_decoded_word}")
            speedup = naive_execution_time / lp_relaxed_execution_time
            print(f"Relaxed LP speedup over naive: {speedup:.2f}x")
        
        # Performance summary
        print(f"\nPerformance Summary:")
        all_correct = []
        if naive_decoded_word == original:
            all_correct.append("Naive")
        if lp_relaxed_decoded_word == original:
            all_correct.append("Relaxed LP")
        
        if all_correct:
            print(f"  Correct decoders: {', '.join(all_correct)}")
        else:
            print(f"  No decoder found the correct codeword!")
        
        print(f"  Fastest method: LP Relaxation" if lp_relaxed_execution_time < naive_execution_time else "Naive")
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        print("Run the code generator first:")
        print("python utils/save_codewords.py")
        print("This will generate LDPC codes in the implementation/codes/ directory")
    except Exception as e:
        print(f"Error in LDPC code test: {e}")
        import traceback
        traceback.print_exc()

def test_multiple_error_patterns():
    """Test with different numbers of errors to see performance degradation."""
    print("\n" + "="*80)
    print("TESTING MULTIPLE ERROR PATTERNS")
    print("="*80)
    
    filename = "./implementation/codes/ldpc_15_11_3_2.pkl"
    
    try:
        codewords, local_constraints, code_info = load_code_data(filename)
        
        # Test with 1, 2, 3 errors
        for num_errors in [1, 2, 3]:
            print(f"\n--- Testing with {num_errors} error(s) ---")
            
            original = codewords[0]  # Use first codeword
            corrupted = corrupt_message(original, num_errors)
            
            print(f"Original:  {original}")
            print(f"Corrupted: {corrupted}")
            print(f"Number of errors: {num_errors}")
            
            # Test naive
            start = time.perf_counter()
            naive_result, _ = findMinimumDistance(corrupted, codewords)
            naive_time = time.perf_counter() - start
            naive_correct = (naive_result == original)
            
            # Test relaxed LP
            start = time.perf_counter()
            lp_result, _ = lp_decode(
                corrupted, codewords, channel_error_prob=0.1,
                relaxation='fundamental', local_constraints=local_constraints
            )
            lp_time = time.perf_counter() - start
            lp_correct = (lp_result == original)
            
            print(f"Naive:     {'✓' if naive_correct else '✗'} ({naive_time:.6f}s)")
            print(f"LP Relax:  {'✓' if lp_correct else '✗'} ({lp_time:.6f}s)")
            print(f"Agree:     {'✓' if naive_result == lp_result else '✗'}")
            
    except Exception as e:
        print(f"Error in multiple error test: {e}")

if __name__ == "__main__":  
    main()
    
    # Uncomment to run additional tests
    # test_multiple_error_patterns()