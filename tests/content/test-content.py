# Example functions with comments

# Function 1: Calculate the sum of two numbers
def add_numbers(a, b):
    """
    Add two numbers and return the result.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum of a and b
    """
    return a + b


# Function 2: Check if a string is a palindrome
def is_palindrome(text):
    """
    Check if a string is a palindrome (reads the same forwards and backwards).
    
    Args:
        text: String to check
    
    Returns:
        True if palindrome, False otherwise
    """
    clean_text = text.lower().replace(" ", "")
    return clean_text == clean_text[::-1]


# Function 3: Find the maximum value in a list
def find_max(numbers):
    """
    Find the maximum value in a list of numbers.
    
    Args:
        numbers: List of numbers
    
    Returns:
        The largest number in the list
    """
    if not numbers:
        return None
    return max(numbers)


# Function 4: Convert temperature from Celsius to Fahrenheit
def celsius_to_fahrenheit(celsius):
    """
    Convert temperature from Celsius to Fahrenheit.
    
    Args:
        celsius: Temperature in Celsius
    
    Returns:
        Temperature in Fahrenheit
    """
    return (celsius * 9/5) + 32