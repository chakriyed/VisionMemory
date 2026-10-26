import json
from argparse import Namespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from vision_memory.memory import Detection
from vision_memory.photos import run_photo, suppress_duplicates


def test_duplicate_suppression_preserves_separate_instances():
    first = Detection('pen', .9, (10, 10, 50, 50), None)
    duplicate = Detection('pen', .7, (11, 11, 51, 51), None)
    second = Detection('pen', .8, (40, 10, 80, 50), None)
    assert suppress_duplicates([first, duplicate, second]) == [first, second]


def test_photo_exports_every_instance_and_original_coordinates(tmp_path):
    image = tmp_path / 'input.png'
    cv2.imwrite(str(image), np.zeros((100, 200, 3), dtype=np.uint8))
    args = Namespace(image=image, device='cpu', open_images_model='trained.pt',
                     classes='pen', model='prompted.pt', output_dir=tmp_path / 'out',
                     confidence=.2)
    trained = Mock(names={0: 'Pen', 1: 'Watch'})
    detections = [Detection('pen', .9, (10, 20, 30, 40), None),
                  Detection('pen', .8, (100, 20, 130, 40), None)]
    with patch('vision_memory.photos.YOLO', return_value=trained), \
         patch('vision_memory.photos.YOLOE') as prompted, \
         patch('vision_memory.photos.predict_robust', return_value=detections) as predict:
        run_photo(args)
    prompted.assert_not_called()
    assert predict.call_args.kwargs['class_ids'] == [0]
    payload = json.loads((args.output_dir / 'input_detections.json').read_text())
    assert len(payload['detections']) == 2
    assert payload['detections'][1]['bbox'] == [100, 20, 130, 40]
    assert payload['width'] == 200
    assert cv2.imread(str(args.output_dir / 'input_annotated.jpg')).shape == (100, 200, 3)


def test_unreadable_image_does_not_load_models(tmp_path):
    args = Namespace(image=tmp_path / 'missing.jpg')
    with patch('vision_memory.photos.YOLO') as loader:
        with pytest.raises(ValueError, match='Could not read image'):
            run_photo(args)
    loader.assert_not_called()
