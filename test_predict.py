"""Small regression checks. Run: python -m unittest test_predict -v"""
import csv
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from predict import character_features, draw_plates, load_models, read_plates, rectify_crop, straighten_preview


SETTINGS = {"plate": {"confidence": 0.5, "image_size": 640},
            "character": {"confidence": 0.25, "image_size": 320}}


# Return deterministic YOLO-shaped results without real weights or GPU inference.
class FakeModel:
    names = {0: "1", 1: "2"}

    def __init__(self, boxes, classes):
        self.boxes = boxes
        self.classes = classes
        self.shape = None

    def predict(self, image, **options):
        self.shape = image.shape
        boxes = SimpleNamespace(xyxy=torch.tensor(self.boxes), cls=torch.tensor(self.classes),
                                conf=torch.tensor([0.9] * len(self.boxes)))
        return [SimpleNamespace(boxes=boxes)]


class PredictionTests(unittest.TestCase):
    def test_characters_are_sorted_inside_the_predicted_crop(self):
        # The reader must receive the predicted crop and return text in spatial reading order.
        detector = FakeModel([[20., 10., 120., 60.]], [0])
        reader = FakeModel([[30., 5., 40., 30.], [5., 5., 15., 30.]], [1, 0])
        result = read_plates(np.zeros((100, 200, 3), np.uint8), detector, reader, SETTINGS)[0]
        self.assertEqual(reader.shape, (50, 100, 3))
        self.assertEqual(result["text"], "12")
        self.assertEqual(result["characters"][0]["box_in_crop"], [5., 5., 15., 30.])

    def test_tiny_crop_boxes_are_scaled_back(self):
        # Tiny crops are enlarged internally; exported character coordinates must be scaled back.
        detector = FakeModel([[20., 10., 29., 15.]], [0])
        reader = FakeModel([[2., 2., 16., 8.]], [0])
        result = read_plates(np.zeros((100, 200, 3), np.uint8), detector, reader, SETTINGS)[0]
        self.assertEqual(reader.shape, (10, 18, 3))
        np.testing.assert_allclose(result["characters"][0]["box_in_crop"], [1, 1, 8, 4])

    def test_hog_shape_and_empty_prediction_display(self):
        # Check the feature contract and the empty-result preview, not model quality.
        features = character_features(np.full((25, 15, 3), 255, np.uint8))
        self.assertEqual(features.shape, (324,))
        self.assertTrue(np.isfinite(features).all())
        self.assertGreater(draw_plates([]).width, 0)

    def test_blank_rectification_abstains(self):
        # A blank image must not produce a fabricated perspective-correction candidate.
        self.assertIsNone(rectify_crop(np.zeros((100, 300, 3), np.uint8)))

    def test_display_rotation_preserves_predictions_and_aligns_centers(self):
        characters = [{"label": "1", "box_in_crop": [10+30*i, 10+6*i, 20+30*i, 30+6*i]} for i in range(3)]
        plate = {"crop": np.zeros((90, 140, 3), np.uint8), "characters": characters, "perspective_applied": False}
        # Display rotation may align boxes but must not mutate the recognition output.
        before = [char["box_in_crop"].copy() for char in characters]
        image, mapped = straighten_preview(plate)
        self.assertEqual([c["box_in_crop"] for c in characters], before)
        self.assertLess(np.ptp([(c["box_in_crop"][1]+c["box_in_crop"][3])/2 for c in mapped]), 1e-4)
        self.assertGreater(image.shape[1], 0)

    def test_changed_weights_are_rejected_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            with open(os.path.join(folder, "plate.pt"), "wb") as stream:
                stream.write(b"changed")
            with open(os.path.join(folder, "settings.csv"), "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["model", "filename", "sha256"])
                writer.writeheader()
                writer.writerows([{"model": "plate", "filename": "plate.pt", "sha256": "incorrect"},
                                  {"model": "character", "filename": "char.pt", "sha256": "incorrect"}])
            # Reject checkpoint/settings mismatches before attempting to deserialize the model.
            with patch("predict.YOLO") as loader, self.assertRaises(ValueError):
                load_models(folder)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
