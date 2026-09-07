import unittest
from scripts.analyze_baseline import stats, summarize


class ErrorAnalysisTests(unittest.TestCase):
    def test_stats_excludes_unresolved_values(self):
        self.assertEqual(stats([None]), {'n': 0})
        self.assertEqual(stats([1, None, 3]), {'n': 2, 'median': 2.0, 'min': 1.0, 'max': 3.0})

    def test_concentration_and_same_label_flips(self):
        template = dict(score=0.0, margin=0.01, inside_ndvi=0.5, ring_ndvi=0.5,
                        inside_brightness=0.1, inside_pixels=5, ring_pixels=50,
                        inside_valid_fraction=1, prior_area=5, gt_area=None,
                        prior_iou=None, gt_coverage_by_prior=None, prior_coverage_by_gt=None)
        rows = [dict(template, key=('a','1','s'), target=0, predicted=True, outcome='fp'),
                dict(template, key=('a','2','s'), target=0, predicted=False, outcome='tn'),
                dict(template, key=('b','1','s'), target=1, predicted=False, outcome='fn')]
        report = summarize(rows)
        self.assertEqual(report['sites'], 2)
        self.assertEqual(report['temporal']['prediction_flip_same_label'], 1)
        self.assertEqual(report['error_concentration']['fp']['sites_with_errors'], 1)
        self.assertEqual(report['bins']['inside_pixels_lt10'], {'n':3, 'errors':2})
        self.assertEqual(report['fp_site_comparison']['tn']['score']['n'], 1)


if __name__ == '__main__':
    unittest.main()
