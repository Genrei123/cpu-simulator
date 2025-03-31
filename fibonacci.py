# fibonacci.py - Fibonacci sequence algorithm for CPU simulation

def fibonacci(n):
    """
    Calculate the nth Fibonacci number
    
    The Fibonacci sequence starts with 0, 1, and each subsequent number
    is the sum of the two preceding numbers (0, 1, 1, 2, 3, 5, 8, 13, ...)
    
    Args:
        n: A positive integer representing the position in the sequence
        
    Returns:
        The nth Fibonacci number
    """
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative indices")
    
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

# The CPU simulator will automatically detect and execute this function
if __name__ == "__main__":
    test_n = 10
    print(f"The {test_n}th Fibonacci number is {fibonacci(test_n)}")