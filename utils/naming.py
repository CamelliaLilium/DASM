"""
Unified naming utilities for result files across all algorithms
"""

def get_result_filename(args):
    """Generate standardized result filename: result_{steg_algorithm}_{embedding_rate}_{test_domains}.txt"""
    
    # Extract embedding rate from dataset_id if available, otherwise use args.embedding_rate
    if hasattr(args, 'dataset_id') and args.dataset_id:
        # Extract from dataset_id like "QIM+PMS+LSB+AHCM_0.5_1s.pkl"
        if '_' in args.dataset_id:
            parts = args.dataset_id.split('_')
            for part in parts:
                try:
                    embedding_rate = float(part)
                    break
                except ValueError:
                    continue
            else:
                embedding_rate = args.embedding_rate
        else:
            embedding_rate = args.embedding_rate
    else:
        embedding_rate = args.embedding_rate
    
    # Clean up test domains (remove spaces, ensure consistent format)
    test_domains = args.test_domains.replace(' ', '').replace(',', '+')
    
    # Generate filename
    filename = f"result_{args.steg_algorithm}_{embedding_rate}_{test_domains}.txt"
    
    return filename