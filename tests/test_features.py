import unittest
from features import extract_features
class FeatureTests(unittest.TestCase):
    def test_url(self):
        x=extract_features(body='Visit https://example.org/path')
        self.assertEqual(x['vector'][0],1)
    def test_sender_mismatch(self):
        x=extract_features(sender='a@example.org',reply_to='a@example.net',body='Hi')
        self.assertEqual(x['vector'][10],1)
    def test_no_link(self):
        x=extract_features(body='Hello team')
        self.assertEqual(x['urls'],[])
if __name__=='__main__':unittest.main()
