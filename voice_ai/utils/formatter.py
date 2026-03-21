import re
from num2words import num2words

def text_to_spoken_words(text):
    """
    Finds all numbers in a string and converts them to spoken English words.
    Example: "aloo ka rate 1295 rupees hai" -> "aloo ka rate one thousand two hundred and ninety-five rupees hai"
    """
    # Find all numbers (including decimals like 1295.50)
    def replace_number(match):
        number_str = match.group(0)
        try:
            # Convert string to float (or int if no decimal)
            number = float(number_str) if '.' in number_str else int(number_str)
            # Use num2words to get the full English pronunciation
            return num2words(number)
        except ValueError:
            return number_str

    # Regex to find numbers in the text
    # \d+ matches digits, (?:\.\d+)? optionally matches a decimal point and more digits
    processed_text = re.sub(r'\d+(?:\.\d+)?', replace_number, text)
    
    # Also replace the ₹ symbol with the word "rupees" just in case
    processed_text = processed_text.replace("₹", " rupees ")
    
    return processed_text