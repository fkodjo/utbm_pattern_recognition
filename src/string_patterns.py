"""
String Pattern Matching Module
Provides functionality for finding and matching patterns in strings
"""

import re
from typing import List, Tuple, Optional


class StringPatternMatcher:
    """
    A class for matching and finding patterns in strings.
    Supports regex patterns, substring matching, and various pattern recognition tasks.
    """

    def __init__(self):
        """Initialize the StringPatternMatcher."""
        pass

    def find_pattern(self, text: str, pattern: str) -> List[int]:
        """
        Find all occurrences of a pattern in text.
        
        Args:
            text: The text to search in
            pattern: The pattern to search for
            
        Returns:
            List of starting indices where pattern is found
        """
        indices = []
        start = 0
        while True:
            index = text.find(pattern, start)
            if index == -1:
                break
            indices.append(index)
            start = index + 1
        return indices

    def match_regex(self, text: str, regex_pattern: str) -> List[str]:
        """
        Find all matches of a regex pattern in text.
        
        Args:
            text: The text to search in
            regex_pattern: The regex pattern to match
            
        Returns:
            List of matched strings
        """
        return re.findall(regex_pattern, text)

    def count_pattern(self, text: str, pattern: str) -> int:
        """
        Count occurrences of a pattern in text.
        
        Args:
            text: The text to search in
            pattern: The pattern to count
            
        Returns:
            Number of occurrences
        """
        return len(self.find_pattern(text, pattern))

    def replace_pattern(self, text: str, pattern: str, replacement: str) -> str:
        """
        Replace all occurrences of a pattern with replacement text.
        
        Args:
            text: The text to modify
            pattern: The pattern to replace
            replacement: The replacement text
            
        Returns:
            Modified text with replacements
        """
        return text.replace(pattern, replacement)

    def extract_emails(self, text: str) -> List[str]:
        """
        Extract email addresses from text.
        
        Args:
            text: The text to search in
            
        Returns:
            List of email addresses found
        """
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return self.match_regex(text, email_pattern)

    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text.
        
        Args:
            text: The text to search in
            
        Returns:
            List of URLs found
        """
        url_pattern = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
        return self.match_regex(text, url_pattern)

    def is_palindrome(self, text: str) -> bool:
        """
        Check if a string is a palindrome.
        
        Args:
            text: The text to check
            
        Returns:
            True if palindrome, False otherwise
        """
        cleaned = re.sub(r'[^a-zA-Z0-9]', '', text.lower())
        return cleaned == cleaned[::-1]

    def find_repeating_substring(self, text: str, min_length: int = 2) -> Optional[str]:
        """
        Find the longest repeating substring in text.
        
        Args:
            text: The text to search in
            min_length: Minimum length of substring to consider
            
        Returns:
            Longest repeating substring or None
        """
        n = len(text)
        longest = None
        max_len = 0

        for i in range(n):
            for j in range(i + min_length, n + 1):
                substring = text[i:j]
                if text.count(substring) > 1 and len(substring) > max_len:
                    longest = substring
                    max_len = len(substring)

        return longest

    def levenshtein_distance(self, str1: str, str2: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Levenshtein distance (number of edits needed)
        """
        if len(str1) < len(str2):
            return self.levenshtein_distance(str2, str1)

        if len(str2) == 0:
            return len(str1)

        previous_row = range(len(str2) + 1)
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]
