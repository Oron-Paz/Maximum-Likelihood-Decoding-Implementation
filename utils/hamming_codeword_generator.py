import numpy as np

def generate_extended_hamming_codewords(r):
    """
    Generate extended Hamming codewords and return as numpy array.
    
    Parameters:
    r (int): Number of check bits (r >= 2)
    
    Returns:
    numpy.ndarray: Array of shape (2^k, n) where each row is a codeword
    
    Examples:
    r=3 -> [8,4,4] code: 16 codewords of length 8
    r=4 -> [16,11,4] code: 2048 codewords of length 16
    """
    
    n = 2**r           
    k = 2**r - r - 1 
    num_codewords = 2**k
    
    print(f"Generating Extended Hamming [{n},{k},4] code...")
    print(f"Creating {num_codewords:,} codewords of length {n}")
    
    codewords = []
    
    for i in range(num_codewords):
        message = [(i >> j) & 1 for j in range(k)]
        
        codeword = create_extended_hamming_codeword(message, r)
        codewords.append(codeword)
        
        if num_codewords > 100 and i % (num_codewords // 20) == 0:
            print(f"Progress: {i/num_codewords*100:.0f}%")
    
    print(f"Generated {len(codewords)} codewords!")
    return np.array(codewords)

def create_extended_hamming_codeword(message_bits, r):
    
    k = len(message_bits)
    n = 2**r
    
    codeword = [0] * n
    
    # Place message bits in positions that are NOT powers of 2 (except last position)
    message_idx = 0
    for pos in range(1, n):  # positions 1 to n-1 (1-indexed)
        if (pos & (pos - 1)) != 0:  # if pos is NOT a power of 2
            codeword[pos - 1] = message_bits[message_idx]  # convert to 0-indexed
            message_idx += 1
    
    # Calculate parity bits for positions that ARE powers of 2
    for i in range(r):
        parity_pos = 2**i  # positions 1, 2, 4, 8, 16, ...
        parity_bit = 0
        
        # XOR all positions that have bit i set in their binary representation
        for pos in range(1, n):
            if pos & parity_pos:  # if bit i is set in pos
                parity_bit ^= codeword[pos - 1]
        
        codeword[parity_pos - 1] = parity_bit  # store parity bit
    
    # Calculate overall parity bit (extension bit at last position)
    overall_parity = sum(codeword[:-1]) % 2
    codeword[n - 1] = overall_parity
    
    return codeword
