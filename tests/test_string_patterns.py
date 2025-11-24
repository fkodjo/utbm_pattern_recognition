"""
Tests for String Pattern Matching Module
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from string_patterns import StringPatternMatcher


class TestStringPatternMatcher(unittest.TestCase):
    """Test cases for StringPatternMatcher class."""

    def setUp(self):
        """Set up test fixtures."""
        self.matcher = StringPatternMatcher()

    def test_find_pattern(self):
        """Test finding pattern occurrences."""
        text = "hello world hello"
        pattern = "hello"
        result = self.matcher.find_pattern(text, pattern)
        self.assertEqual(result, [0, 12])

    def test_find_pattern_no_match(self):
        """Test finding pattern with no matches."""
        text = "hello world"
        pattern = "goodbye"
        result = self.matcher.find_pattern(text, pattern)
        self.assertEqual(result, [])

    def test_count_pattern(self):
        """Test counting pattern occurrences."""
        text = "abc abc abc"
        pattern = "abc"
        count = self.matcher.count_pattern(text, pattern)
        self.assertEqual(count, 3)

    def test_replace_pattern(self):
        """Test replacing patterns."""
        text = "hello world hello"
        result = self.matcher.replace_pattern(text, "hello", "hi")
        self.assertEqual(result, "hi world hi")

    def test_extract_emails(self):
        """Test email extraction."""
        text = "Contact us at info@example.com or support@test.org"
        emails = self.matcher.extract_emails(text)
        self.assertEqual(len(emails), 2)
        self.assertIn("info@example.com", emails)
        self.assertIn("support@test.org", emails)

    def test_extract_urls(self):
        """Test URL extraction."""
        text = "Visit https://example.com or http://test.org for more info"
        urls = self.matcher.extract_urls(text)
        self.assertEqual(len(urls), 2)

    def test_is_palindrome_true(self):
        """Test palindrome detection - positive case."""
        self.assertTrue(self.matcher.is_palindrome("racecar"))
        self.assertTrue(self.matcher.is_palindrome("A man a plan a canal Panama"))

    def test_is_palindrome_false(self):
        """Test palindrome detection - negative case."""
        self.assertFalse(self.matcher.is_palindrome("hello"))
        self.assertFalse(self.matcher.is_palindrome("world"))

    def test_find_repeating_substring(self):
        """Test finding repeating substrings."""
        text = "abcabc"
        result = self.matcher.find_repeating_substring(text)
        self.assertIn(result, ["abc", "ab", "bc"])

    def test_levenshtein_distance(self):
        """Test Levenshtein distance calculation."""
        distance = self.matcher.levenshtein_distance("kitten", "sitting")
        self.assertEqual(distance, 3)
        
        distance = self.matcher.levenshtein_distance("hello", "hello")
        self.assertEqual(distance, 0)


if __name__ == '__main__':
    unittest.main()
