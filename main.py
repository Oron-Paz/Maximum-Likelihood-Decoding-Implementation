from src.naive_solver.naive import findMinimumDistance
from utils.save_codewords import load_codewords
from src.lp_solver.linear_programming import lp_decode
from utils.corrupt import corrupt_message
import pickle
import time


def main():
    filename = "./codes/hamming_32_26_4_partial_10000k.pkl"      # Large: ~10M codewords
    try:
        codewords = load_codewords(filename)
        codewords_list = [codeword.tolist() for codeword in codewords]
        codewords_set = set(tuple(cw) for cw in codewords_list)

        message = codewords_list[523428]
        print(f"Message to be sent: {message}")

        corrupted_message = corrupt_message(message, 1)
        while tuple(corrupted_message) in codewords_set:
            corrupted_message = corrupt_message(message,1)

        print(f"Corrupted Message:  {corrupted_message}")
        if corrupted_message in codewords_list:
            print(f"CORRUPTED MESSAGE IS VALID CODEWORD, DECODING WILL FAIL") # shold never reach this line
        
        
        print("\nML Decoding with Pre-loaded Codewords")
        print("=" * 45)
        
        
        #print(f"Corrupted message: {corrupted_message}")
        print(f"Message length: {len(corrupted_message)}")
        print(f"Searching through {len(codewords_list):,} codewords...")
        
        # Perform Naive ML decoding and time it
        print("\n1. NAIVE ML DECODING:")
        print("-" * 30)
        start_time = time.time()
        best_codeword_naive, min_distance = findMinimumDistance(corrupted_message, codewords_list, withStopLoss=False)
        decode_time_naive = time.time() - start_time
        
        print(f"Decoding completed in {decode_time_naive:.2f} seconds")
        print(f"Decoded codeword:  {best_codeword_naive}")
        print(f"Hamming distance:  {min_distance}")
        
        # Perform Paper's LP ML decoding and time it
        print("\n2. PAPER'S LINEAR PROGRAMMING ML DECODING:")
        print("-" * 30)
        start_time = time.time()
        best_codeword_lp, optimal_cost = lp_decode(corrupted_message, channel_error_prob=0.1, r=5)
        decode_time_lp = time.time() - start_time
        
        print(f"Decoding completed in {decode_time_lp:.2f} seconds")
        print(f"Decoded codeword:  {best_codeword_lp}")
        print(f"Optimal cost:      {optimal_cost:.6f}")
        
        # Comparison
        print("\n3. COMPARISON:")
        print("-" * 30)
        print(f"Results match:     {best_codeword_naive == best_codeword_lp}")
        print(f"Naive time:        {decode_time_naive:.2f} seconds")
        print(f"LP time:           {decode_time_lp:.2f} seconds")
        if decode_time_lp > 0:
            print(f"Speedup:           {decode_time_naive / decode_time_lp:.1f}x")
        
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        print("Run the codeword generator first to create the codeword files.")
        print("Example: python src/save_codewords.py")

if __name__ == "__main__":
    main()