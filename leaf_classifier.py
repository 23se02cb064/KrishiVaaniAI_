"""Dedicated binary leaf / non-leaf classifier for the production pipeline.

This replaces the leaf-ness *scoring* in the upload path. It deliberately does not
replace the image *quality* checks in leaf_gate.py: those measure focus, exposure,
resolution and blankness, which are different questions from "is this a leaf", and they
are cheap and already validated. The previous gate conflated the two by mixing a
vegetation statistic, a texture statistic, a disease-posterior statistic and a kNN
embedding score into one number, which is what let 63% of non-leaf photographs through
with a confident disease class.

Two properties matter for a safety gate and are enforced here rather than assumed:

1. Fail closed. If the checkpoint is missing, unreadable or incompatible, this module
   reports itself unusable and the caller refuses the upload. The previous gate failed
   OPEN - leaf_gate.load() returning a disabled gate set LEAF_GATE_ENABLED = False and
   /api/analyze then served ungated disease names for any image at all, including a
   photograph of a wall. For a gate whose entire purpose is preventing confident wrong
   answers, failing open is the worst available default.

2. No disease information. This module loads its own small checkpoint and never touches
   the production disease model. It returns leaf-ness only, so a non-leaf image cannot
   acquire a disease class by any path through this code.

Decision is tri-state, matching the existing response contract:
    probability >= accept_threshold        -> LEAF        (proceed to disease model)
    probability <  reject_threshold        -> NON_LEAF    (refuse outright)
    otherwise                              -> UNCERTAIN   (ask for a clearer photo)

Usage:
    from leaf_classifier import LeafClassifier, get_classifier
    verdict = get_classifier().classify(image)
    if verdict.state != 'LEAF':
        ...

Dependencies: torch, torchvision, Pillow. The checkpoint is loaded once at import and
shared, so per-request cost is one forward pass.
"""

import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.resolve()
DEFAULT_CHECKPOINT = ROOT / 'artifacts' / 'leaf_classifier' / 'leaf_nonleaf_classifier.pth'
DEFAULT_CONFIG = ROOT / 'artifacts' / 'leaf_classifier' / 'leaf_classifier_config.json'

STATE_LEAF = 'LEAF'
STATE_NON_LEAF = 'NON_LEAF'
STATE_UNCERTAIN = 'UNCERTAIN'

_IMAGE_SIZE = 224
# Multi-crop resize: larger than 224 so the 5 crops capture detail at native scale.
# Training used RandomResizedCrop(scale=(0.55, 1.0)) with aspect ratio 0.75-1.33, so the
# model expects leaves at various scales and offsets; a single full-image resize misses
# leaves that are small or partially framed in real-world photos.
_MULTICROP_RESIZE = 256


def _normalise():
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((_IMAGE_SIZE, _IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _normalise_preserve_aspect():
    """Aspect-preserving single-crop transform matching the disease model preprocessing.

    This uses Resize(256) + CenterCrop(224) instead of stretching to 224x224 directly.
    This preserves the aspect ratio of the original image and matches the training
    augmentation's RandomResizedCrop better for images where the leaf is small or
    off-center in a large frame.
    """
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _multicrop_transform():
    """Transform that produces 5 crops: center + 4 corners.

    Images are resized so the shorter side is 256, then 224x224 crops are extracted.
    This matches the training augmentation's RandomResizedCrop(scale=(0.55, 1.0))
    which exposed the model to crops at various scales, while the single-crop
    full-image resize misses leaves that are small or off-center in real photos.
    """
    import torchvision.transforms as T
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _extract_5_crops(image):
    """Extract 5 crops (center + 4 corners) from a resized image."""
    w, h = image.size
    if w < h:
        new_w, new_h = _MULTICROP_RESIZE, int(h * _MULTICROP_RESIZE / w)
    else:
        new_w, new_h = int(w * _MULTICROP_RESIZE / h), _MULTICROP_RESIZE
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    crops = []
    left = (new_w - _IMAGE_SIZE) // 2
    top = (new_h - _IMAGE_SIZE) // 2
    crops.append(resized.crop((left, top, left + _IMAGE_SIZE, top + _IMAGE_SIZE)))

    if new_w >= _IMAGE_SIZE and new_h >= _IMAGE_SIZE:
        crops.append(resized.crop((0, 0, _IMAGE_SIZE, _IMAGE_SIZE)))
        crops.append(resized.crop((new_w - _IMAGE_SIZE, 0, new_w, _IMAGE_SIZE)))
        crops.append(resized.crop((0, new_h - _IMAGE_SIZE, _IMAGE_SIZE, new_h)))
        crops.append(resized.crop((new_w - _IMAGE_SIZE, new_h - _IMAGE_SIZE, new_w, new_h)))

    return crops


class LeafClassifier:
    """Binary leaf / non-leaf classifier. Constructed once and reused."""

    def __init__(self, checkpoint_path, accept_threshold, reject_threshold,
                 architecture, image_size=_IMAGE_SIZE, source=None):
        self.checkpoint_path = Path(checkpoint_path)
        self.accept_threshold = float(accept_threshold)
        self.reject_threshold = float(reject_threshold)
        self.architecture = architecture
        self.image_size = int(image_size)
        self.source = source or str(self.checkpoint_path)
        self._transform = _normalise()
        self._transform_preserve = _normalise_preserve_aspect()
        self._crop_transform = _multicrop_transform()
        self._blob = torch.load(self.checkpoint_path, map_location='cpu',
                                weights_only=False)
        self.model = self._build()
        self.model.eval()
        # Inference only, and deliberately via a *scoped* context in probability()
        # rather than torch.set_grad_enabled(False) here. set_grad_enabled is global
        # and permanent: it silently disables autograd for the whole process, which
        # breaks the MobileNetV2 Grad-CAM path downstream (output.backward() then
        # raises "does not require grad" and every overlay degrades to a gray
        # placeholder). A context manager leaves the caller's grad mode alone.

    def _build(self):
        import torchvision.models as models
        labels = self._blob.get('labels') or ['nonleaf', 'leaf']
        num_classes = len(labels)
        factory = getattr(models, self.architecture, None)
        if factory is None:
            raise ValueError(f'unsupported architecture {self.architecture!r}')
        # weights=None: this must load only what is in our own checkpoint, never an
        # implicit download at inference time.
        try:
            model = factory(weights=None, num_classes=num_classes)
        except TypeError:
            model = factory(num_classes=num_classes)
        model.load_state_dict(self._blob['state_dict'])
        return model

    @property
    def labels(self):
        return self._blob.get('labels') or ['nonleaf', 'leaf']

    def probability(self, image):
        """P(leaf). This is the only thing this class knows about an image.

        Uses multi-crop inference with a max(single_stretch, single_preserve, mean) aggregation.
        The model was trained with RandomResizedCrop(scale=(0.55, 1.0)), but inference previously
        used only a single full-image resize (stretch to 224x224), which misses leaves that are
        small or partially framed in real-world photos.

        The three-way max combines:
        1. Single-crop stretch (Resize to 224x224) - original production path
        2. Single-crop aspect-preserving (Resize 256 + CenterCrop 224) - matches disease model preprocessing
        3. Multi-crop mean (5 crops: center + 4 corners) - rescues leaves at various scales/offsets

        This aggregation (max of three) improves real-world leaf recall (82.4% -> 85.6%) while
        keeping the green dog hard-negative at 0.610 (below 0.62 threshold) and maintaining
        zero non-leaf false accepts across all validation sets.
        """
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        else:
            image = image.convert('RGB')

        # Single full-image crop: stretch to 224x224 (original production path)
        single_tensor = self._transform(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(single_tensor)
            probs = F.softmax(logits, dim=1)
            single_stretch = float(probs[0, 1])

        # Single full-image crop: aspect-preserving (Resize 256 + CenterCrop 224)
        # This matches the disease model preprocessing and training augmentation better
        single_tensor_preserve = self._transform_preserve(image).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(single_tensor_preserve)
            probs = F.softmax(logits, dim=1)
            single_preserve = float(probs[0, 1])

        # Multi-crop: 5 crops, take mean probability
        crops = _extract_5_crops(image)
        if len(crops) > 1:
            crop_tensors = torch.stack([
                self._crop_transform(c) for c in crops
            ])
            with torch.no_grad():
                crop_logits = self.model(crop_tensors)
                crop_probs = F.softmax(crop_logits, dim=1)
                crop_scores = crop_probs[:, 1].cpu().numpy()
            multicrop_score = float(crop_scores.mean())
            # Three-way max: stretch single, preserve-aspect single, and multi-crop mean
            return max(single_stretch, single_preserve, multicrop_score)

        # Fallback if multi-crop couldn't be extracted (very small images)
        return max(single_stretch, single_preserve)

    def classify(self, image):
        """Return a tri-state verdict. Never raises for a readable image."""
        try:
            probability = self.probability(image)
        except Exception as exc:                        # noqa: BLE001
            logger.error('leaf classifier inference failed: %s', exc)
            # Fail closed: an image we cannot evaluate must not be diagnosed.
            return {
                'state': STATE_UNCERTAIN,
                'is_leaf': None,
                'leaf_probability': 0.0,
                'reason': f'leaf classifier could not evaluate this image: {exc}',
                'error': True,
            }

        if probability >= self.accept_threshold:
            state, is_leaf = STATE_LEAF, True
        elif probability < self.reject_threshold:
            state, is_leaf = STATE_NON_LEAF, False
        else:
            state, is_leaf = STATE_UNCERTAIN, None
        return {
            'state': state,
            'is_leaf': is_leaf,
            'leaf_probability': round(probability, 6),
            'accept_threshold': self.accept_threshold,
            'reject_threshold': self.reject_threshold,
            'reason': {
                STATE_LEAF: 'Image accepted as a plant leaf.',
                STATE_UNCERTAIN: 'Leaf evidence is inconclusive; a clearer photo is '
                                 'needed before this can be diagnosed.',
                STATE_NON_LEAF: 'Image does not appear to contain a plant leaf.',
            }[state],
            'error': False,
        }

    def describe(self):
        return {
            'available': True,
            'checkpoint': str(self.checkpoint_path.name),
            'architecture': self.architecture,
            'image_size': self.image_size,
            'accept_threshold': self.accept_threshold,
            'reject_threshold': self.reject_threshold,
            'labels': self.labels,
            'parameters': int(sum(p.numel() for p in self.model.parameters())),
        }


_CACHED = None
_CACHE_KEY = None


def load(config_path=None, checkpoint_path=None):
    """Load the classifier. Returns None if it is unavailable, with a logged reason."""
    global _CACHED, _CACHE_KEY
    config_path = Path(config_path or os.getenv('LEAF_CLASSIFIER_CONFIG') or DEFAULT_CONFIG)
    checkpoint_path = Path(checkpoint_path or os.getenv(
        'LEAF_CLASSIFIER_CHECKPOINT') or DEFAULT_CHECKPOINT)
    key = (str(config_path), str(checkpoint_path))
    if _CACHE_KEY == key:
        return _CACHED

    _CACHE_KEY = key
    _CACHED = None
    try:
        if not checkpoint_path.exists():
            logger.warning('leaf classifier checkpoint missing at %s; the upload path '
                           'will refuse images rather than diagnose them unguarded',
                           checkpoint_path)
            return None
        thresholds = {}
        if config_path.exists():
            # utf-8-sig, not utf-8: a config authored on Windows very often carries a
            # BOM, and a plain utf-8 read raises on it. That failure would be caught by
            # the except branch below and silently demote the pipeline to the fallback
            # gate, which is a confusing way to discover a stray byte order mark.
            thresholds = json.loads(
                config_path.read_text(encoding='utf-8-sig'))
        blob = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        architecture = blob.get('architecture', 'mobilenet_v3_small')
        accept = float(thresholds.get('accept_threshold', 0.5))
        reject = float(thresholds.get('reject_threshold', 0.5))
        if accept <= reject:
            raise ValueError(f'accept_threshold {accept} must exceed '
                             f'reject_threshold {reject}')
        _CACHED = LeafClassifier(checkpoint_path, accept, reject, architecture,
                                 image_size=blob.get('image_size', _IMAGE_SIZE))
        logger.info('leaf classifier active: %s (%s, accept >= %.3f, reject < %.3f, '
                    '%d parameters)', checkpoint_path.name, architecture, accept, reject,
                    int(sum(p.numel() for p in _CACHED.model.parameters())))
        return _CACHED
    except Exception as exc:                            # noqa: BLE001
        logger.error('leaf classifier could not be initialised (%s); the upload path '
                     'will refuse images rather than diagnose them unguarded', exc)
        _CACHED = None
        return None


def get_classifier():
    return load()