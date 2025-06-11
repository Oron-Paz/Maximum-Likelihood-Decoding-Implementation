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
    print("TEST 2: Extended Hamming [32,26,4] Code (Partial)")
    print("-" * 45)
    
    filename = "./codes/hamming_32_26_4_partial_10000k.pkl"
    
    try:
        codewords = load_codewords(filename)
        codewords_list = [codeword.tolist() for codeword in codewords]
        codewords_set = set(tuple(cw) for cw in codewords_list)

        original = codewords_list[523428]
        corrupted = corrupt_message(original, 1)
        
        while tuple(corrupted) in codewords_set:
            corrupted = corrupt_message(original, 1)

        print(f"Original:  {original}")
        print(f"Corrupted: {corrupted}")
        print(f"Codebook size: {len(codewords_list):,} codewords")
        
        print("\nStarting Naive decoder:")
        print("-" * 45)
        start = time.perf_counter()
        naive_decoded_word , hamming_distance = findMinimumDistance(corrupted, codewords_list)
        end = time.perf_counter()
        naive_execution_time = end - start

        print(f"Decoded word: {naive_decoded_word}")
        print(f"Hamming Distance {hamming_distance}")
        print(f"Execution time: {naive_execution_time}s")
       
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        print("Run the codeword generator first:")
        print("python utils/save_codewords.py")
    except Exception as e:
        print(f"Error in large code test: {e}")

if __name__ == "__main__":
    main()