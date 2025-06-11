import numpy as np
import pickle
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hamming_codeword_generator import generate_extended_hamming_codewords, create_extended_hamming_codeword

def save_codewords(r, filename=None):
    """Generate and save codewords to file"""
    if filename is None:
        n = 2**r
        k = 2**r - r - 1
        filename = f"codes/hamming_{n}_{k}_4.pkl"
    
    print(f"Generating Extended Hamming code with r={r}...")
    codewords = generate_extended_hamming_codewords(r)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save as pickle file for faster laoding
    with open(filename, 'wb') as f:
        pickle.dump(codewords, f)
    
    print(f"Saved {len(codewords)} codewords to {filename}")
    print(f"File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
    
    return filename

def load_codewords(filename):
    """Load codewords from file"""
    print(f"Loading codewords from {filename}...")
    
    with open(filename, 'rb') as f:
        codewords = pickle.load(f)
    
    print(f"Loaded {len(codewords)} codewords of length {len(codewords[0])}")
    return codewords

def save_partial_codewords(r, max_codewords, filename=None):
    """Generate and save only the first max_codewords from the code"""
    if filename is None:
        n = 2**r
        k = 2**r - r - 1
        filename = f"codes/hamming_{n}_{k}_4_partial_{max_codewords//1000}k.pkl"
    
    n = 2**r           
    k = 2**r - r - 1 
    total_possible = 2**k
    
    print(f"Generating PARTIAL Extended Hamming [{n},{k},4] code...")
    print(f"Creating {max_codewords:,} out of {total_possible:,} possible codewords")
    
    codewords = []
    
    for i in range(min(max_codewords, total_possible)):
        # Generate message bits from counter i
        message = [(i >> j) & 1 for j in range(k)]
        
        # Create codeword using your existing function
        codeword = create_extended_hamming_codeword(message, r)
        codewords.append(codeword)
        
        # Progress indicator
        if i > 0 and i % (max_codewords // 20) == 0:
            print(f"Progress: {i:,} / {max_codewords:,} ({i/max_codewords*100:.0f}%)")
    
    # Convert to numpy array
    codewords_array = np.array(codewords)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save as pickle file
    with open(filename, 'wb') as f:
        pickle.dump(codewords_array, f)
    
    print(f"\nSaved {len(codewords)} codewords to {filename}")
    print(f"File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
    
    return filename

if __name__ == "__main__":
    print("Extended Hamming Code Generator")
    print("="*50)
    print("Options:")
    print("1. Small [8,4,4] - 16 codewords")
    print("2. Medium [16,11,4] - 2,048 codewords") 
    print("3. Large [32,26,4] - 10 million codewords (partial)")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        save_codewords(3)  # r=3 -> [8,4,4]
    elif choice == "2":
        save_codewords(4)  # r=4 -> [16,11,4]
    elif choice == "3":
        # Generate 10 million codewords from [32,26,4] code
        save_partial_codewords(5, 10_000_000)  # r=5, 10M codewords
    else:
        print("Invalid choice!")