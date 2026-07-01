import unittest
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from parsing.resume_parser import clean_resume_text
from features.text_features import create_tfidf_features

class TestFeatures(unittest.TestCase):
    def test_clean_resume_text(self):
        raw_text = "Hello World! Check this link: http://example.com #awesome @user"
        expected = "hello world check this link"
        self.assertEqual(clean_resume_text(raw_text), expected)

    def test_tfidf_creation(self):
        texts = ["python developer", "java developer", "data scientist"]
        matrix, vectorizer = create_tfidf_features(texts)
        self.assertEqual(matrix.shape[0], 3)
        self.assertTrue(len(vectorizer.get_feature_names_out()) > 0)

if __name__ == '__main__':
    unittest.main()
