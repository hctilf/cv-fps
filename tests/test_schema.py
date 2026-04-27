"""Test schema: validate JSON output packet structure against expected schema."""

import json
import unittest

import jsonschema

from src.postprocess import postprocess

DETECTION_SCHEMA = {
    "type": "object",
    "required": ["frame_id", "ts_infer_ms", "detections"],
    "properties": {
        "frame_id": {"type": "integer"},
        "ts_infer_ms": {"type": "number"},
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["class", "bbox", "conf"],
                "properties": {
                    "class": {"type": "string"},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "conf": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
            },
        },
    },
}


class TestSchema(unittest.TestCase):
    def test_empty_detections_schema(self):
        """Validate schema for frame with no detections."""
        result = postprocess(raw_results=None, frame_id=1, infer_ms=4.5)
        jsonschema.validate(instance=result, schema=DETECTION_SCHEMA)
        self.assertEqual(result["frame_id"], 1)
        self.assertEqual(result["detections"], [])

    def test_valid_detections_schema(self):
        """Validate schema for frame with detections (mocked)."""
        # Simulate a result with detections
        result = {
            "frame_id": 42,
            "ts_infer_ms": 5.2,
            "detections": [
                {
                    "class": "EnemySoldier",
                    "bbox": [10.0, 20.0, 100.0, 200.0],
                    "conf": 0.94,
                },
                {
                    "class": "HealthPack",
                    "bbox": [300.0, 400.0, 350.0, 450.0],
                    "conf": 0.72,
                },
            ],
        }
        jsonschema.validate(instance=result, schema=DETECTION_SCHEMA)

    def test_invalid_conf_range(self):
        """Detect schema violation for confidence out of range."""
        result = {
            "frame_id": 1,
            "ts_infer_ms": 3.0,
            "detections": [
                {"class": "Weapon", "bbox": [0, 0, 50, 50], "conf": 1.5},
            ],
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=result, schema=DETECTION_SCHEMA)

    def test_invalid_bbox_length(self):
        """Detect schema violation for bbox with wrong number of elements."""
        result = {
            "frame_id": 1,
            "ts_infer_ms": 3.0,
            "detections": [
                {"class": "Vehicle", "bbox": [0, 0, 50], "conf": 0.8},
            ],
        }
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.validate(instance=result, schema=DETECTION_SCHEMA)

    def test_all_valid_classes(self):
        """Validate all known class names produce valid output."""
        valid_classes = [
            "EnemySoldier",
            "AllySoldier",
            "Weapon",
            "Vehicle",
            "HealthPack",
        ]
        for cls in valid_classes:
            result = {
                "frame_id": 1,
                "ts_infer_ms": 3.0,
                "detections": [
                    {"class": cls, "bbox": [0.0, 0.0, 100.0, 100.0], "conf": 0.85}
                ],
            }
            jsonschema.validate(instance=result, schema=DETECTION_SCHEMA)


if __name__ == "__main__":
    unittest.main()
