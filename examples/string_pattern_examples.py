"""
Examples of using the String Pattern Matching Module
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from string_patterns import StringPatternMatcher


def main():
    """Run string pattern matching examples."""
    print("=" * 60)
    print("STRING PATTERN MATCHING EXAMPLES")
    print("=" * 60)

    matcher = StringPatternMatcher()

    # Example 1: Finding patterns
    print("\n1. Finding pattern occurrences:")
    text = "The cat sat on the mat. The cat was fat."
    pattern = "cat"
    indices = matcher.find_pattern(text, pattern)
    print(f"   Text: '{text}'")
    print(f"   Pattern: '{pattern}'")
    print(f"   Found at indices: {indices}")

    # Example 2: Counting patterns
    print("\n2. Counting patterns:")
    count = matcher.count_pattern(text, "at")
    print(f"   Pattern 'at' appears {count} times in the text")

    # Example 3: Extracting emails
    print("\n3. Extracting email addresses:")
    email_text = "Contact support@example.com or sales@company.org for help"
    emails = matcher.extract_emails(email_text)
    print(f"   Text: '{email_text}'")
    print(f"   Found emails: {emails}")

    # Example 4: Extracting URLs
    print("\n4. Extracting URLs:")
    url_text = "Visit https://example.com and http://test.org for more"
    urls = matcher.extract_urls(url_text)
    print(f"   Text: '{url_text}'")
    print(f"   Found URLs: {urls}")

    # Example 5: Palindrome detection
    print("\n5. Palindrome detection:")
    test_words = ["racecar", "hello", "madam", "python"]
    for word in test_words:
        is_palindrome = matcher.is_palindrome(word)
        print(f"   '{word}' is palindrome: {is_palindrome}")

    # Example 6: Finding repeating substrings
    print("\n6. Finding repeating substrings:")
    repeat_text = "abcabcabc"
    repeating = matcher.find_repeating_substring(repeat_text)
    print(f"   Text: '{repeat_text}'")
    print(f"   Longest repeating substring: '{repeating}'")

    # Example 7: Levenshtein distance
    print("\n7. Levenshtein distance (edit distance):")
    pairs = [("kitten", "sitting"), ("hello", "hallo"), ("python", "python")]
    for word1, word2 in pairs:
        distance = matcher.levenshtein_distance(word1, word2)
        print(f"   Distance between '{word1}' and '{word2}': {distance}")

    # Example 8: Pattern replacement
    print("\n8. Pattern replacement:")
    original = "I like cats. Cats are great pets."
    replaced = matcher.replace_pattern(original, "cats", "dogs")
    print(f"   Original: '{original}'")
    print(f"   Replaced: '{replaced}'")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
