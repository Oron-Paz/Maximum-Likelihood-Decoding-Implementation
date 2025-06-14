import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull
import itertools

def lp_decode(received_message, codewords_list, channel_error_prob=0.1, relaxation='exact', local_constraints=None):
    """
    LP decoder with different relaxation methods.
    
    Args:
        received_message: received vector
        codewords_list: list of valid codewords  
        channel_error_prob: channel crossover probability
        relaxation: 'exact' or 'fundamental'
        local_constraints: list of local constraint specifications for fundamental relaxation
    """
    received = np.array(received_message)
    n = len(received)
    
    gamma = compute_gamma(received, channel_error_prob)
    print(f"Gamma (objective): {gamma}")
    
    if relaxation == 'exact':
        return lp_decode_exact(received, codewords_list, gamma)
    elif relaxation == 'fundamental':
        if local_constraints is None:
            raise ValueError("local_constraints required for fundamental relaxation")
        return lp_decode_fundamental_relaxation(received, codewords_list, gamma, local_constraints)
    else:
        raise ValueError(f"Unknown relaxation: {relaxation}")

def lp_decode_exact(received, codewords_list, gamma):
    """Exact LP decoding using the full codeword polytope."""
    codewords = np.array(codewords_list)
    
    hull = ConvexHull(codewords)  # ConvexHull from scipy returns equations that define the polytope shapes used below
    A_ub = hull.equations[:, :-1]  # Normal vectors
    b_ub = -hull.equations[:, -1]  # Constants
    
    print(f"ConvexHull: {len(A_ub)} facet constraints")
    
    # Solve LP: min γᵀx subject to Ax ≤ b
    # https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html 
    # Check out above link for info on what's happening below, but basically A_ub = coefficients of constraints as a matrix
    # b_ub = the actual value we're constrained on as a vector to match each row in A
    result = linprog(method='highs', c=gamma, A_ub=A_ub, b_ub=b_ub)

    if not result.success:
        print(f"LP failed: {result.message}")
        return None, float('inf')
    
    print(f"\nLP solution x: {result.x}")
    print(f"LP optimal cost: {result.fun:.6f}")
    
    # Find closest codeword to LP solution
    distances = np.sum((codewords - result.x)**2, axis=1)
    closest_idx = np.argmin(distances)
    best_codeword = codewords[closest_idx]
    
    print(f"LP solution distance to codeword: {np.sqrt(distances[closest_idx]):.6f}")
    print(f"Closest codeword: {best_codeword}")
    
    return best_codeword.tolist(), result.fun

def lp_decode_fundamental_relaxation(received, codewords_list, gamma, local_constraints):
    """
    Fundamental relaxation using intersection of local constraint polytopes.
    This implements the relaxation described in: https://arxiv.org/pdf/cs/0602087
    
    Args:
        received: received vector
        codewords_list: list of valid codewords (for final rounding)
        gamma: objective function coefficients
        local_constraints: list of dicts, each containing:
            - 'variables': indices of variables involved in this constraint
            - 'valid_patterns': list of valid local patterns for these variables
    """
    codewords = np.array(codewords_list)
    n = len(gamma)
    
    print(f"Fundamental relaxation with {len(local_constraints)} local constraints")
    
    # Collect all constraints from local polytopes
    A_ub_list = []
    b_ub_list = []
    
    for i, constraint in enumerate(local_constraints):
        variables = constraint['variables']
        valid_patterns = np.array(constraint['valid_patterns'])
        
        print(f"Local constraint {i}: variables {variables}, {len(valid_patterns)} patterns")
        
        if len(valid_patterns) <= 1:
            continue  # Skip trivial constraints
            
        # Create convex hull of local patterns
        try:
            if len(valid_patterns) == 2 and valid_patterns.shape[1] == 1:
                # Special case for 1D constraints
                min_val = np.min(valid_patterns)
                max_val = np.max(valid_patterns)
                A_local = np.array([[-1], [1]])
                b_local = np.array([-min_val, max_val])
            else:
                hull = ConvexHull(valid_patterns)
                A_local = hull.equations[:, :-1]
                b_local = -hull.equations[:, -1]
        except Exception as e:
            print(f"Warning: ConvexHull failed for constraint {i}: {e}")
            continue
        
        # Embed local constraints into global space
        A_global = np.zeros((A_local.shape[0], n))
        for j, var_idx in enumerate(variables):
            if var_idx < n:  # Safety check
                A_global[:, var_idx] = A_local[:, j]
        
        A_ub_list.append(A_global)
        b_ub_list.append(b_local)
    
    if not A_ub_list:
        print("Warning: No valid local constraints, falling back to box constraints")
        # Fallback to simple box constraints [0,1]^n
        A_ub = np.vstack([np.eye(n), -np.eye(n)])
        b_ub = np.hstack([np.ones(n), np.zeros(n)])
    else:
        A_ub = np.vstack(A_ub_list)
        b_ub = np.hstack(b_ub_list)
    
    print(f"Total constraints: {len(b_ub)}")
    
    # Add box constraints [0,1] for each variable (this is the fundamental polytope P)
    bounds = [(0, 1) for _ in range(n)]
    
    # Solve LP: min γᵀx subject to x ∈ ∩_{j∈J} conv(C_j) and x ∈ [0,1]^n
    result = linprog(method='highs', c=gamma, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

    if not result.success:
        print(f"LP failed: {result.message}")
        return None, float('inf')
    
    print(f"\nFundamental LP solution x: {result.x}")
    print(f"LP optimal cost: {result.fun:.6f}")
    
    # Find closest codeword to LP solution
    distances = np.sum((codewords - result.x)**2, axis=1)
    closest_idx = np.argmin(distances)
    best_codeword = codewords[closest_idx]
    
    print(f"LP solution distance to codeword: {np.sqrt(distances[closest_idx]):.6f}")
    print(f"Closest codeword: {best_codeword}")
    
    return best_codeword.tolist(), result.fun

def compute_gamma(received, crossover_prob):
    """
    Compute log-likelihood ratios for binary symmetric channel.
    For BSC with crossover probability p:
    - If received bit is 0, we want to favor x_i = 0 
    - If received bit is 1, we want to favor x_i = 1
    """
    received_array = np.array(received)
    p = crossover_prob
    
    # Avoid log(0) issues
    p = max(min(p, 0.999), 0.001)
    
    gamma = np.zeros(len(received_array))
    
    for i in range(len(received_array)):
        if received_array[i] == 0:
            # Favor x_i = 0, so cost is positive when x_i = 1
            gamma[i] = np.log((1-p) / p)  
        else:  
            # Favor x_i = 1, so cost is negative when x_i = 1
            gamma[i] = np.log(p / (1-p))  
    
    return gamma

def lp_decode_box_relaxation(received_message, codewords_list, channel_error_prob=0.1):
    """
    Ultra-simple relaxation: just use box constraints [0,1]^n.
    This is the most relaxed polytope possible.
    """
    received = np.array(received_message)
    n = len(received)
    gamma = compute_gamma(received, channel_error_prob)
    
    print(f"Box relaxation: just [0,1]^n constraints")
    
    # Only box constraints: 0 ≤ x_i ≤ 1 for all i
    bounds = [(0, 1) for _ in range(n)]
    
    # Solve LP: min γᵀx subject to x ∈ [0,1]^n
    result = linprog(method='highs', c=gamma, bounds=bounds)
    
    if not result.success:
        print(f"LP failed: {result.message}")
        return None, float('inf')
    
    print(f"Box LP solution x: {result.x}")
    print(f"LP optimal cost: {result.fun:.6f}")
    
    # Find closest codeword
    codewords = np.array(codewords_list)
    distances = np.sum((codewords - result.x)**2, axis=1)
    closest_idx = np.argmin(distances)
    best_codeword = codewords[closest_idx]
    
    print(f"LP solution distance to codeword: {np.sqrt(distances[closest_idx]):.6f}")
    print(f"Closest codeword: {best_codeword}")
    
    return best_codeword.tolist(), result.fun

def lp_decode_simple_parity_relaxation(received_message, codewords_list, channel_error_prob=0.1, 
                                       local_constraints=None):
    """
    Simple relaxation: only use parity check constraints, ignore complex local patterns.
    """
    received = np.array(received_message)
    n = len(received)
    gamma = compute_gamma(received, channel_error_prob)
    
    print(f"Simple parity relaxation")
    
    # Extract simple parity constraints from local_constraints
    A_ub_list = []
    b_ub_list = []
    
    if local_constraints:
        for constraint in local_constraints:
            variables = constraint['variables']
            
            # Simple parity constraint: sum of variables should be even
            # This means: sum(x_i for i in variables) ≤ |variables| - 0.5
            # and: sum(x_i for i in variables) ≥ 0.5
            # But we'll use even simpler: sum ≤ |variables| and sum ≥ 0
            
            # Upper bound: sum(x_i) ≤ len(variables)
            A_upper = np.zeros(n)
            for var_idx in variables:
                if var_idx < n:
                    A_upper[var_idx] = 1
            A_ub_list.append(A_upper)
            b_ub_list.append(len(variables))
            
            # Could add lower bound, but let's keep it simple
    
    if A_ub_list:
        A_ub = np.vstack(A_ub_list)
        b_ub = np.array(b_ub_list)
        print(f"Using {len(b_ub)} simple parity constraints")
    else:
        A_ub = None
        b_ub = None
        print("No constraints - falling back to box relaxation")
    
    # Box constraints
    bounds = [(0, 1) for _ in range(n)]
    
    # Solve LP
    result = linprog(method='highs', c=gamma, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    
    if not result.success:
        print(f"LP failed: {result.message}")
        return None, float('inf')
    
    print(f"Simple LP solution x: {result.x}")
    print(f"LP optimal cost: {result.fun:.6f}")
    
    # Find closest codeword
    codewords = np.array(codewords_list)
    distances = np.sum((codewords - result.x)**2, axis=1)
    closest_idx = np.argmin(distances)
    best_codeword = codewords[closest_idx]
    
    print(f"LP solution distance to codeword: {np.sqrt(distances[closest_idx]):.6f}")
    print(f"Closest codeword: {best_codeword}")
    
    return best_codeword.tolist(), result.fun

def lp_decode_subset_relaxation(received_message, codewords_list, channel_error_prob=0.1, 
                                local_constraints=None, max_constraints=10):
    """
    Use only a subset of local constraints to keep the problem manageable.
    """
    received = np.array(received_message)
    n = len(received)
    gamma = compute_gamma(received, channel_error_prob)
    
    print(f"Subset relaxation using max {max_constraints} constraints")
    
    # Use only first few local constraints
    if local_constraints and len(local_constraints) > max_constraints:
        local_constraints = local_constraints[:max_constraints]
        print(f"Reduced from many constraints to {len(local_constraints)}")
    
    # Collect constraints but limit the number of patterns
    A_ub_list = []
    b_ub_list = []
    
    if local_constraints:
        for i, constraint in enumerate(local_constraints):
            variables = constraint['variables']
            valid_patterns = constraint['valid_patterns']
            
            # Limit to first 16 patterns to avoid explosion
            valid_patterns = valid_patterns[:16]
            
            print(f"Constraint {i}: variables {variables}, using {len(valid_patterns)} patterns")
            
            if len(valid_patterns) <= 1:
                continue
            
            # Create convex hull of limited patterns
            try:
                if len(valid_patterns) == 2 and len(variables) == 1:
                    # Special case for 1D
                    min_val = min(p[0] for p in valid_patterns)
                    max_val = max(p[0] for p in valid_patterns)
                    A_local = np.array([[-1], [1]])
                    b_local = np.array([-min_val, max_val])
                else:
                    hull = ConvexHull(valid_patterns)
                    A_local = hull.equations[:, :-1]
                    b_local = -hull.equations[:, -1]
            except Exception as e:
                print(f"Skipping constraint {i}: {e}")
                continue
            
            # Embed into global space
            A_global = np.zeros((A_local.shape[0], n))
            for j, var_idx in enumerate(variables):
                if var_idx < n:
                    A_global[:, var_idx] = A_local[:, j]
            
            A_ub_list.append(A_global)
            b_ub_list.append(b_local)
    
    if A_ub_list:
        A_ub = np.vstack(A_ub_list)
        b_ub = np.hstack(b_ub_list)
        print(f"Total constraints: {len(b_ub)} (much better!)")
    else:
        A_ub = None
        b_ub = None
    
    # Box constraints
    bounds = [(0, 1) for _ in range(n)]
    
    # Solve LP
    result = linprog(method='highs', c=gamma, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    
    if not result.success:
        print(f"LP failed: {result.message}")
        return None, float('inf')
    
    print(f"Subset LP solution x: {result.x}")
    print(f"LP optimal cost: {result.fun:.6f}")
    
    # Find closest codeword
    codewords = np.array(codewords_list)
    distances = np.sum((codewords - result.x)**2, axis=1)
    closest_idx = np.argmin(distances)
    best_codeword = codewords[closest_idx]
    
    print(f"LP solution distance to codeword: {np.sqrt(distances[closest_idx]):.6f}")
    print(f"Closest codeword: {best_codeword}")
    
    return best_codeword.tolist(), result.fun

def lp_decode_random_sampling_relaxation(received_message, codewords_list, channel_error_prob=0.1, 
                                         num_sample_codewords=100):
    """
    Use only a random sample of codewords to define the polytope.
    This is like the "core" relaxation approach.
    """
    received = np.array(received_message)
    n = len(received)
    gamma = compute_gamma(received, channel_error_prob)
    
    print(f"Random sampling relaxation using {num_sample_codewords} codewords")
    
    # Sample a subset of codewords
    if len(codewords_list) > num_sample_codewords:
        indices = np.random.choice(len(codewords_list), size=num_sample_codewords, replace=False)
        sample_codewords = [codewords_list[i] for i in indices]
    else:
        sample_codewords = codewords_list
    
    sample_codewords = np.array(sample_codewords)
    print(f"Using {len(sample_codewords)} sampled codewords")
    
    # Create convex hull of sampled codewords
    try:
        hull = ConvexHull(sample_codewords)
        A_ub = hull.equations[:, :-1]
        b_ub = -hull.equations[:, -1]
        print(f"Sampled ConvexHull: {len(A_ub)} constraints")
    except Exception as e:
        print(f"ConvexHull failed: {e}, falling back to box relaxation")
        return lp_decode_box_relaxation(received_message, codewords_list, channel_error_prob)
    
    # Solve LP
    result = linprog(method='highs', c=gamma, A_ub=A_ub, b_ub=b_ub)
    
    if not result.success:
        print(f"LP failed: {result.message}")
        return None, float('inf')
    
    print(f"Sampling LP solution x: {result.x}")
    print(f"LP optimal cost: {result.fun:.6f}")
    
    # Find closest codeword from FULL codebook
    codewords = np.array(codewords_list)
    distances = np.sum((codewords - result.x)**2, axis=1)
    closest_idx = np.argmin(distances)
    best_codeword = codewords[closest_idx]
    
    print(f"LP solution distance to codeword: {np.sqrt(distances[closest_idx]):.6f}")
    print(f"Closest codeword: {best_codeword}")
    
    return best_codeword.tolist(), result.fun

def load_code_from_file(filename):
    """
    Load code data from saved pickle file.
    
    Args:
        filename: path to pickle file containing code data
        
    Returns:
        codewords, local_constraints, code_info
    """
    import pickle
    
    with open(filename, 'rb') as f:
        code_data = pickle.load(f)
    
    codewords = code_data['codewords']
    local_constraints = code_data['local_constraints']
    code_info = code_data['code_info']
    
    print(f"Loaded {code_info['code_type']} code:")
    print(f"  n={code_info['n']}, k={code_info['k']}")
    print(f"  {len(codewords)} codewords, {len(local_constraints)} local constraints")
    
    return codewords, local_constraints, code_info

def test_lp_decoding(filename, received_message, channel_error_prob=0.1, compare_methods=True):
    """
    Test LP decoding methods on a specific received message.
    
    Args:
        filename: path to saved code file
        received_message: received vector to decode
        channel_error_prob: channel crossover probability
        compare_methods: whether to compare exact vs fundamental methods
    """
    import time
    
    # Load code
    codewords, local_constraints, code_info = load_code_from_file(filename)
    
    print(f"\n{'='*60}")
    print(f"TESTING LP DECODING")
    print(f"{'='*60}")
    print(f"Code: {code_info['code_type']}, n={code_info['n']}, k={code_info['k']}")
    print(f"Received: {received_message}")
    print(f"Channel error prob: {channel_error_prob}")
    
    results = {}
    
    # Test exact LP decoding
    if compare_methods:
        print(f"\n--- EXACT LP DECODING ---")
        try:
            start_time = time.time()
            exact_result, exact_cost = lp_decode(
                received_message, codewords, channel_error_prob, relaxation='exact'
            )
            exact_time = time.time() - start_time
            
            results['exact'] = {
                'decoded': exact_result,
                'cost': exact_cost,
                'time': exact_time,
                'success': True
            }
            print(f"Exact LP result: {exact_result}")
            print(f"Decode time: {exact_time:.6f}s")
            
        except Exception as e:
            print(f"Exact LP failed: {e}")
            results['exact'] = {'success': False, 'error': str(e)}
    
    # Test fundamental relaxation
    print(f"\n--- FUNDAMENTAL RELAXATION ---")
    try:
        start_time = time.time()
        fund_result, fund_cost = lp_decode(
            received_message, codewords, channel_error_prob, 
            relaxation='fundamental', local_constraints=local_constraints
        )
        fund_time = time.time() - start_time
        
        results['fundamental'] = {
            'decoded': fund_result,
            'cost': fund_cost,
            'time': fund_time,
            'success': True
        }
        print(f"Fundamental result: {fund_result}")
        print(f"Decode time: {fund_time:.6f}s")
        
    except Exception as e:
        print(f"Fundamental relaxation failed: {e}")
        results['fundamental'] = {'success': False, 'error': str(e)}
    
    # Compare results
    if compare_methods and results.get('exact', {}).get('success') and results.get('fundamental', {}).get('success'):
        print(f"\n--- COMPARISON ---")
        exact_decoded = results['exact']['decoded']
        fund_decoded = results['fundamental']['decoded']
        
        if exact_decoded == fund_decoded:
            print("✓ Both methods produced the same result")
        else:
            print("✗ Methods produced different results:")
            print(f"  Exact:       {exact_decoded}")
            print(f"  Fundamental: {fund_decoded}")
        
        speedup = results['exact']['time'] / results['fundamental']['time']
        print(f"Fundamental relaxation speedup: {speedup:.2f}x")
    
    return results

def batch_test_decoding(filename, num_tests=10, channel_error_prob=0.1):
    """
    Run batch testing of LP decoding methods.
    
    Args:
        filename: path to saved code file
        num_tests: number of random tests to run
        channel_error_prob: channel crossover probability
    """
    import time
    import random
    
    # Load code
    codewords, local_constraints, code_info = load_code_from_file(filename)
    
    print(f"\n{'='*60}")
    print(f"BATCH TESTING LP DECODING")
    print(f"{'='*60}")
    print(f"Code: {code_info['code_type']}, n={code_info['n']}, k={code_info['k']}")
    print(f"Number of tests: {num_tests}")
    print(f"Channel error prob: {channel_error_prob}")
    
    results = {
        'exact': {'times': [], 'successes': 0, 'total': 0},
        'fundamental': {'times': [], 'successes': 0, 'total': 0},
        'agreements': 0
    }
    
    np.random.seed(42)  # For reproducibility
    
    for test_num in range(num_tests):
        # Generate random test case
        true_codeword = random.choice(codewords)
        n = len(true_codeword)
        
        # Add random errors
        error_pattern = np.random.binomial(1, channel_error_prob, n)
        received = (np.array(true_codeword) + error_pattern) % 2
        
        print(f"\nTest {test_num + 1}/{num_tests}")
        print(f"True codeword: {true_codeword}")
        print(f"Received:      {received.tolist()}")
        
        gamma = compute_gamma(received, channel_error_prob)
        
        # Test exact method (skip if too slow for large codes)
        if code_info['n'] <= 15:  # Only test exact for small codes
            try:
                start_time = time.time()
                exact_result, _ = lp_decode_exact(received, codewords, gamma)
                exact_time = time.time() - start_time
                
                results['exact']['times'].append(exact_time)
                results['exact']['total'] += 1
                if exact_result == true_codeword:
                    results['exact']['successes'] += 1
                
                print(f"Exact: {exact_result} ({'✓' if exact_result == true_codeword else '✗'})")
                
            except Exception as e:
                print(f"Exact failed: {e}")
                results['exact']['total'] += 1
        
        # Test fundamental method
        try:
            start_time = time.time()
            fund_result, _ = lp_decode_fundamental_relaxation(received, codewords, gamma, local_constraints)
            fund_time = time.time() - start_time
            
            results['fundamental']['times'].append(fund_time)
            results['fundamental']['total'] += 1
            if fund_result == true_codeword:
                results['fundamental']['successes'] += 1
            
            print(f"Fundamental: {fund_result} ({'✓' if fund_result == true_codeword else '✗'})")
            
            # Check agreement (only if both methods ran)
            if (code_info['n'] <= 15 and results['exact']['total'] == results['fundamental']['total'] 
                and 'exact_result' in locals() and exact_result == fund_result):
                results['agreements'] += 1
                
        except Exception as e:
            print(f"Fundamental failed: {e}")
            results['fundamental']['total'] += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH TEST SUMMARY")
    print(f"{'='*60}")
    
    for method in ['exact', 'fundamental']:
        data = results[method]
        if data['total'] > 0:
            success_rate = data['successes'] / data['total']
            avg_time = np.mean(data['times']) if data['times'] else 0
            
            print(f"\n{method.upper()} METHOD:")
            print(f"  Success rate: {success_rate:.2%} ({data['successes']}/{data['total']})")
            print(f"  Average time: {avg_time:.6f}s")
    
    if results['exact']['total'] > 0 and results['fundamental']['total'] > 0:
        agreement_rate = results['agreements'] / min(results['exact']['total'], results['fundamental']['total'])
        print(f"\nAgreement rate: {agreement_rate:.2%}")
        
        if len(results['exact']['times']) > 0 and len(results['fundamental']['times']) > 0:
            speedup = np.mean(results['exact']['times']) / np.mean(results['fundamental']['times'])
            print(f"Average speedup: {speedup:.2f}x")
    
    return results

if __name__ == "__main__":
    # Example usage
    print("Testing LP relaxation decoder...")
    
    # You would use this with your saved codes:
    # test_lp_decoding("codes/ldpc_8_4_3_2.pkl", [1, 0, 1, 0, 1, 0, 1, 0])
    # batch_test_decoding("codes/ldpc_15_11_3_2.pkl", num_tests=20)
    
    print("To use this decoder:")
    print("1. First run the code_saver.py to generate and save codes")
    print("2. Then use test_lp_decoding() or batch_test_decoding() with the saved files")
    print("3. Example: test_lp_decoding('codes/ldpc_8_4_3_2.pkl', [1,0,1,0,1,0,1,0])")