from src.naive_solver.naive import findMinimumDistance
from utils.save_codewords import load_codewords
from src.lp_solver.linear_programming import lp_decode
from utils.corrupt import corrupt_message
import time

def main():
    print("Testing Paper's LP Approach vs Naive Method")
    print("\n" + "="*60)
    test_large_code()

def test_large_code():
    print("TEST: Extended Hamming [32,26,4] Code (Partial)")
    print("-" * 45)
    
    filename = "./codes/hamming_32_26_4_partial_10000k.pkl"
    
    try:
        codewords = load_codewords(filename)
        codewords_list = [codeword.tolist() for codeword in codewords]
        codewords_set = set(tuple(cw) for cw in codewords_list)
        print("Done loading codewords now creating corrupted message...\n")

        original = codewords_list[523428]
        corrupted = corrupt_message(original, 1)
        
        while tuple(corrupted) in codewords_set:
            corrupted = corrupt_message(original, 1)

        print(f"Original:  {original}")
        print(f"Corrupted: {corrupted}")
        print(f"Codebook size: {len(codewords_list):,} codewords")
        
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
        
        # Test 2: LP decoder (exact)
        print("\n" + "="*60)
        print("LP DECODER (exact):")
        print("-" * 45)
        
        start = time.perf_counter()
        lp_decoded_word, lp_cost = lp_decode(corrupted, codewords_list, channel_error_prob=0.1, relaxation='exact')
        end = time.perf_counter()
        lp_execution_time = end - start
        
        print(f"Decoded word: {lp_decoded_word}")
        print(f"LP cost: {lp_cost:.6f}")
        print(f"Execution time: {lp_execution_time:.3f}s")
        print(f"Correct decoding: {lp_decoded_word == original}")
        
        # Comparison
        print("\n" + "="*60)
        print("COMPARISON:")
        print("-" * 45)
        print(f"Both methods agree: {naive_decoded_word == lp_decoded_word}")
        print(f"Naive time:  {naive_execution_time:.3f}s")
        print(f"LP time:     {lp_execution_time:.3f}s")
        if lp_execution_time > 0:
            speedup = naive_execution_time / lp_execution_time
            print(f"Speedup:     {speedup:.2f}x {'(LP faster)' if speedup > 1 else '(Naive faster)'}")
        

        
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        print("Run the codeword generator first:")
        print("python utils/save_codewords.py")
    except Exception as e:
        print(f"Error in large code test: {e}")

if __name__ == "__main__":
    # First test small code where both methods work
    print("\n" + "="*80 + "\n")
    
    # Then test large code to show why relaxations are needed
    main()