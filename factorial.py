# factorial.py - Example algorithm for CPU simulation

def factorial(n):
    """
    Calculate the factorial of n (n!)
    
    Args:
        n: A positive integer
        
    Returns:
        The factorial of n
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    
    result = 1
    for i in range(1, n + 1):
        result *= i
    
    return result

# The CPU simulator will automatically detect and execute this function
# You can test this file independently:
if __name__ == "__main__":
    test_n = 5
    print(f"Factorial of {test_n} is {factorial(test_n)}")