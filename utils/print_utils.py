import time

def print_decoder_result(method_name, decoded_word, cost_or_distance, execution_time, 
                        original_word, corrupted_word=None, is_cost=True):
    """
    Print standardized decoder results.
    
    Args:
        method_name: Name of the decoding method (e.g., "NAIVE DECODER", "LP DECODER (box relaxation)")
        decoded_word: The decoded codeword
        cost_or_distance: Either LP cost or Hamming distance
        execution_time: Time taken in seconds
        original_word: The original uncorrupted word for comparison
        corrupted_word: The corrupted input (optional, for extra checks)
        is_cost: True if cost_or_distance is LP cost, False if Hamming distance
    """
    print("\n" + "="*60)
    print(f"{method_name}:")
    print("-" * 45)
    
    print(f"Decoded word: {decoded_word}")
    
    if is_cost:
        print(f"LP cost: {cost_or_distance:.6f}")
    else:
        print(f"Hamming Distance: {cost_or_distance}")
    
    print(f"Execution time: {execution_time:.6f}s")
    print(f"Correct decoding: {decoded_word == original_word}")
    
    if corrupted_word is not None:
        print(f"Is the same as corrupted message: {decoded_word == corrupted_word}")

def run_decoder_test(decoder_func, method_name, corrupted_word, original_word, 
                    codewords=None, **kwargs):
    """
    Run a decoder test and print results using standardized format.
    
    Args:
        decoder_func: The decoder function to test
        method_name: Display name for the method
        corrupted_word: The corrupted input
        original_word: Original word for comparison
        codewords: List of codewords (for naive decoder)
        **kwargs: Additional arguments to pass to decoder_func
    
    Returns:
        tuple: (decoded_word, cost_or_distance, execution_time)
    """
    start = time.perf_counter()
    
    if codewords is not None:
        # For naive decoder that needs codewords as first argument
        result = decoder_func(corrupted_word, codewords, **kwargs)
    else:
        # For LP decoders
        result = decoder_func(corrupted_word, **kwargs)
    
    end = time.perf_counter()
    execution_time = end - start
    
    # Handle different return formats
    if isinstance(result, tuple) and len(result) >= 2:
        decoded_word, cost_or_distance = result[0], result[1]
    else:
        decoded_word, cost_or_distance = result, 0.0
    
    # Determine if this is cost or distance based on method name
    is_cost = "NAIVE" not in method_name.upper()
    
    print_decoder_result(
        method_name=method_name,
        decoded_word=decoded_word,
        cost_or_distance=cost_or_distance,
        execution_time=execution_time,
        original_word=original_word,
        corrupted_word=corrupted_word,
        is_cost=is_cost
    )
    
    return decoded_word, cost_or_distance, execution_time

def print_test_setup(original, corrupted, codewords, code_info):
    """Print the test setup information."""
    print("Testing Paper's LP Approach vs Naive Method")
    print("\n" + "="*60)
    print("TEST: Naive VS Different Relaxation Techniques:")
    print("-" * 45)
    
    print(f"Original:  {original}")
    print(f"Corrupted: {corrupted}")
    print(f"Codebook size: {len(codewords):,} codewords")
    print(f"Code parameters: n={code_info['n']}, k={code_info['k']}, rate={code_info['k']/code_info['n']:.3f}")

def print_comparison_summary(results):
    """
    Print a comparison summary of all decoder results.
    
    Args:
        results: List of tuples (method_name, decoded_word, cost_or_distance, execution_time, correct)
    """
    print("\n" + "="*60)
    print("COMPARISON SUMMARY:")
    print("-" * 45)
    
    # Find fastest method
    fastest_method = min(results, key=lambda x: x[3])
    
    # Count correct decoders
    correct_methods = [r[0] for r in results if r[4]]
    
    print(f"Correct decoders: {', '.join(correct_methods) if correct_methods else 'None'}")
    print(f"Fastest method: {fastest_method[0]} ({fastest_method[3]:.6f}s)")
    
    # Performance table
    print(f"\n{'Method':<30} | {'Correct':<8} | {'Time (s)':<10}")
    print("-" * 52)
    for method, _, _, exec_time, correct in results:
        status = "✓" if correct else "✗"
        print(f"{method:<30} | {status:<8} | {exec_time:<10.6f}")