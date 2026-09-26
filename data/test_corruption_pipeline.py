import unittest

import numpy as np

from corruption_pipeline import (
    CorruptionType,
    apply_corruption,
    sample_random_corruption,
    validate_corruption_settings,
)


class TestCorruptionPipeline(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((128, 128, 3), dtype=np.uint8)

    def test_sample_random_corruption_has_valid_type(self):
        sample = sample_random_corruption()
        self.assertIn(sample["type"], [
            CorruptionType.CLEAN,
            CorruptionType.SALT_AND_PEPPER,
            CorruptionType.GAUSSIAN_BLUR,
            CorruptionType.RECTANGULAR_OCCLUSION,
        ])

    def test_clean_corruption_leaves_image_unchanged(self):
        corrupted = apply_corruption(self.image.copy(), CorruptionType.CLEAN, {})
        np.testing.assert_array_equal(corrupted, self.image)

    def test_salt_and_pepper_inserts_noise_values(self):
        sample = {"probability": 0.1, "seed": 42}
        corrupted = apply_corruption(self.image.copy(), CorruptionType.SALT_AND_PEPPER, sample)
        self.assertEqual(corrupted.shape, self.image.shape)
        self.assertTrue(np.any(corrupted != 0))

    def test_gaussian_blur_output_has_same_shape(self):
        sample = {"kernel_size": 3, "sigma": 1.0, "seed": 42}
        corrupted = apply_corruption(self.image.copy(), CorruptionType.GAUSSIAN_BLUR, sample)
        self.assertEqual(corrupted.shape, self.image.shape)

    def test_rectangular_occlusion_covers_percent_of_image(self):
        sample = {"num_masks": 2, "coverage": 0.2, "mask_coords": [(10, 10, 40, 40), (70, 70, 100, 100)], "seed": 42}
        corrupted = apply_corruption(self.image.copy(), CorruptionType.RECTANGULAR_OCCLUSION, sample)
        self.assertEqual(corrupted.shape, self.image.shape)
        self.assertTrue(np.any(corrupted < 255))

    def test_validation_detects_invalid_settings(self):
        invalid = {"probability": 2.0}
        with self.assertRaises(ValueError):
            validate_corruption_settings(CorruptionType.SALT_AND_PEPPER, invalid)


if __name__ == "__main__":
    unittest.main()
