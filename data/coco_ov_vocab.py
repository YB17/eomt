"""COCO open-vocabulary class names and prompt templates."""

from __future__ import annotations

from typing import Dict, List

COCO_THINGS_80: List[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

COCO_STUFF_53: List[str] = [
    "banner",
    "blanket",
    "branch",
    "bridge",
    "building-other",
    "bush",
    "cabinet",
    "cage",
    "cardboard",
    "carpet",
    "ceiling-other",
    "ceiling-tile",
    "cloth",
    "clothes",
    "clouds",
    "counter",
    "cupboard",
    "curtain",
    "desk-stuff",
    "dirt",
    "door-stuff",
    "fence",
    "floor-marble",
    "floor-other",
    "floor-stone",
    "floor-tile",
    "floor-wood",
    "flower",
    "fog",
    "food-other",
    "fruit",
    "furniture-other",
    "grass",
    "gravel",
    "ground-other",
    "hill",
    "house",
    "leaves",
    "light",
    "metal",
    "mirror-stuff",
    "moss",
    "mountain",
    "mud",
    "napkin",
    "net",
    "paper",
    "pavement",
    "plaster",
    "plastic",
    "platform",
    "playingfield",
    "railing",
]

TEMPLATES_THINGS: List[str] = [
    "a photo of a {}.",
    "a {} in the scene.",
    "a detailed depiction of a {}.",
]

TEMPLATES_STUFF: List[str] = [
    "a patch of {}.",
    "the {} background.",
    "a texture of {}.",
]

TEMPLATES_THINGS_ZH: List[str] = [
    "一张关于{}的照片。",
    "场景中的{}。",
]

TEMPLATES_STUFF_ZH: List[str] = [
    "一块{}纹理。",
    "{}的背景。",
]

SYNONYMS: Dict[str, List[str]] = {
    "cell phone": ["mobile phone", "smartphone"],
    "couch": ["sofa"],
    "tv": ["television"],
    "airplane": ["aeroplane", "plane"],
    "motorcycle": ["motorbike"],
    "traffic light": ["stoplight"],
    "microwave": ["microwave oven"],
    "refrigerator": ["fridge"],
    "hair drier": ["hair dryer"],
    "building-other": ["building"],
    "ceiling-other": ["ceiling"],
    "floor-other": ["floor"],
}


def build_templates(multilingual: bool = False) -> Dict[str, List[str]]:
    if multilingual:
        return {
            "things": list(TEMPLATES_THINGS) + list(TEMPLATES_THINGS_ZH),
            "stuff": list(TEMPLATES_STUFF) + list(TEMPLATES_STUFF_ZH),
        }
    return {"things": list(TEMPLATES_THINGS), "stuff": list(TEMPLATES_STUFF)}


OPEN_VOCAB_UNSEEN_THINGS_20: List[str] = [
    "airplane",
    "bus",
    "train",
    "boat",
    "bench",
    "bird",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "couch",
    "microwave",
    "oven",
    "sink",
    "toothbrush",
    "motorcycle",
]

OPEN_VOCAB_UNSEEN_STUFF_17: List[str] = [
    "banner",
    "blanket",
    "bridge",
    "ceiling-other",
    "ceiling-tile",
    "cloth",
    "clouds",
    "counter",
    "curtain",
    "fence",
    "floor-stone",
    "ground-other",
    "hill",
    "light",
    "mirror-stuff",
    "pavement",
    "platform",
]

OPEN_VOCAB_SPLITS: Dict[str, Dict[str, List[str]]] = {
    "ovp_val": {
        "seen_things": [c for c in COCO_THINGS_80 if c not in OPEN_VOCAB_UNSEEN_THINGS_20],
        "unseen_things": list(OPEN_VOCAB_UNSEEN_THINGS_20),
        "seen_stuff": [c for c in COCO_STUFF_53 if c not in OPEN_VOCAB_UNSEEN_STUFF_17],
        "unseen_stuff": list(OPEN_VOCAB_UNSEEN_STUFF_17),
    }
}


def get_open_vocab_split(name: str) -> Dict[str, List[str]]:
    """Return a dictionary containing seen/unseen splits for the given split name."""

    if name not in OPEN_VOCAB_SPLITS:
        raise KeyError(f"Unknown open-vocabulary split: {name}")
    split = OPEN_VOCAB_SPLITS[name]
    return {
        "seen_things": list(split["seen_things"]),
        "unseen_things": list(split["unseen_things"]),
        "seen_stuff": list(split["seen_stuff"]),
        "unseen_stuff": list(split["unseen_stuff"]),
    }
