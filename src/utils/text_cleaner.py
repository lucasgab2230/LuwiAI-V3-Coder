import re

def normalize_whitespace(text: str) -> str:
    """
    Normalizes whitespace in the input string.

    This function replaces multiple consecutive spaces or tabs with a single space,
    replaces multiple consecutive newlines with a single newline, and removes
    leading/trailing whitespace from the entire string.

    Args:
        text: The input string with potentially inconsistent whitespace.

    Returns:
        The string with normalized whitespace.
    """
    # Replace multiple spaces with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Strip leading and trailing whitespace
    text = text.strip() # Remove whitespace from the beginning and end of the string
    return text

# Add more cleaning functions as needed for technical text
# def remove_special_chars(text: str) -> str:
#     """
#     Removes or handles special characters that might interfere with processing.
#     (Implementation details would depend on specific characters to target)
#     """
#     pass
