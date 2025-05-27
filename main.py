from src.naive_solver.naive import findMinimumDistance
import pickle
import time

def load_codewords(filename):
    """Load codewords from pickle file"""
    print(f"Loading codewords from {filename}...")
    start_time = time.time()
    
    with open(filename, 'rb') as f:
        codewords = pickle.load(f)
    
    load_time = time.time() - start_time
    print(f"Loaded {len(codewords)} codewords in {load_time:.2f} seconds")
    return codewords

def main():
    filename = "codes/hamming_32_26_4_partial_10000k.pkl"      # Large: ~10M codewords
    
    try:
        # Load codewords from file (much faster than generating!)
        codewords = load_codewords(filename)
        codewords_list = [codeword.tolist() for codeword in codewords]
        
        print("\nML Decoding with Pre-loaded Codewords")
        print("=" * 45)
        
        # Your corrupted message here (adjust length based on code)
        if "32_26" in filename:
            # [32,26,4] code - 32 bit message
            corrupted_message = [0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0,
                               1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]
            # [32,26,4] code - 32 bit message
            corrupted_message = [0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0,
                               1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]
        elif "16_11" in filename:
            # [16,11,4] code - 16 bit message
            corrupted_message = [0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0]
        else:
            # [8,4,4] code - 8 bit message
            corrupted_message = [0, 1, 0, 0, 1, 0, 1, 1]
        
        print(f"Corrupted message: {corrupted_message}")
        print(f"Message length: {len(corrupted_message)}")
        print(f"Searching through {len(codewords_list):,} codewords...")
        
        # Perform ML decoding and time it
        start_time = time.time()
        best_codeword, min_distance = findMinimumDistance(corrupted_message, codewords_list)
        decode_time = time.time() - start_time
        
        print(f"\nDecoding completed in {decode_time:.2f} seconds")
        print(f"Decoded codeword:  {best_codeword}")
        print(f"Hamming distance:  {min_distance}")
        
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        print("Run the codeword generator first to create the codeword files.")
        print("Example: python src/codeword_saver.py")

if __name__ == "__main__":
    main()