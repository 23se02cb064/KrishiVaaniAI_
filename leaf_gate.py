"""Leaf-suitability gate: decide whether an image plausibly shows a plant leaf.

Why this exists
---------------
The production model is a 38-class leaf-disease classifier. It was trained only on
leaf images, but nothing stops a user from submitting a photo of a face, a laptop or
a wall. Such an image still produces logits, so the model returns a confident disease
name. Measured on the frozen 60-image external set it already returns 13.33% top-1
accuracy at 54.05% mean confidence, and a non-leaf photo is confidently wrong by
construction. Confidence alone therefore cannot be used to detect non-leaf input: the
model is *most* confident precisely when it is least correct on out-of-domain images.

How the gate works
------------------
Four leaf-likeness terms, deliberately chosen to fail on different inputs, plus a
separate hard quality gate that runs first. The weights are the ones measured by
experiments/calibrate_leaf_gate.py over 457 leaf photographs and 24 non-leaf
photographs; they are not chosen by intuition.

Term A - solid vegetation structure (weight 0.35)
    The share of the frame occupied by vegetation that survives two erosions, i.e. the
    area of a contiguous green blob rather than a green texture. Measured: leaves 0.38
    and 0.40 median, non-leaves 0.02. Structure rather than raw greenness is the point,
    because grass, foliage and flat green paint are green but not solid, and because a
    diseased leaf can be almost entirely brown.

    It can never accept an image on its own, and it can never reject one on its own
    either: a dried, necrotic or all-yellow leaf scores zero here and must still be
    rescued by the other terms.

Term B - feature-manifold distance (weight 0.00, measured and deliberately disabled)
    The 1280-d pooled penultimate features compared against reference leaf embeddings
    built from the PlantVillage training split, scored as the mean cosine distance to
    the k nearest references. This was the original rescue path for brown leaves, but
    measurement rejected it: the two worst non-leaf photographs sit 0.140 from the
    prototype set, closer than most real field leaves, so the term rewarded them.
    It is still computed and reported, and `weights` can re-enable it, but the
    calibrated default is zero.

Term C - class coherence (weight 0.30)
    A photograph of a leaf produces a coherent 38-way posterior: the mass concentrates
    on one crop (Apple___, Tomato___, ...) and the top class stands clearly ahead of
    the runner-up. On an out-of-domain image the mass spreads across crops and the
    margin collapses. This uses the shape of the distribution, never the confidence
    value, so it cannot be fooled the way a confidence threshold is.

    Measured weakness, documented rather than hidden: on *field* photographs the
    production model is nearly as confused as on non-leaf photographs (13.07%
    external accuracy), so this signal is weak exactly where it is most needed.

Term D - photographic detail (weight 0.35)
    The fraction of the frame with no local gradient. Measured: leaf photographs 0.04
    and 0.21 median, non-leaves 0.61. Wallpapers, smooth gradients and abstract renders
    are nearly flat, while a leaf photograph carries detail everywhere. The bound is
    deliberately lenient because an out-of-focus field photograph is legitimately flat
    and must not be refused for it.

Quality gate (hard, runs first)
    Minimum resolution, extreme blur, near-black or blown-out exposure, blank/solid
    frames and undecodable files are rejected outright, because no classifier can say
    anything useful about them. Exposure is judged on the *subject* rather than the
    frame: a leaf on a white studio background is a good photograph, and counting that
    background as clipped highlights used to refuse it. Thresholds are deliberately
    permissive on darkness so that genuinely dark or necrotic leaves survive.

    Every one of the 457 measured leaf images passes the quality gate, and all 38
    classes are accepted 8 out of 8.

Calibrated behaviour (recalibrated 2026-10-05, audit/expanded_signals.json)
-----------------------------------------------------------------------------
    accept >= 0.9000, reject < 0.0500
    weights         vegetation 0.7, feature 0.3, coherence 0.0, detail 0.0
    measured here   studio leaves    438 of 456 diagnosed (96%), 0 refused
                    real-world leaves  71 of 153 diagnosed (46%), 0 refused
                    non-leaf           0 of  27 given a diagnosis

    The previous calibration (accept >= 0.6500, weights coherence 0.3 / detail 0.35 /
    feature 0.0) was wrong and is corrected here. It admitted 17 of 27 non-leaf
    photographs with a confident disease class. The reason is structural rather than a
    bad threshold: coherence scores how peaked the 38-way disease posterior is, so a
    confidently wrong prediction scores a perfect 1.0, and detail is 1.0 for almost any
    photograph. Those two signals carried 0.65 of the weight between them, so an image
    with no vegetation at all scored 0.35*0 + 0.3*1 + 0.35*1 = 0.65 exactly, landing on
    the accept boundary. Raising the threshold alone would not have helped much: nine
    non-leaf images were pinned at exactly 0.65 and five remain admitted at 0.70.

    Two limitations are worth stating plainly. First, the kNN `feature` term cannot
    separate real-world leaves from non-leaf photographs on this backbone - at a distance
    cut that keeps 95% of real leaves, 21 of 27 non-leaf images are admitted too, and
    rebuilding the prototype cloud with real-world leaves does not fix that. Second, only
    46% of real-world leaf photographs clear the gate at this threshold. Both point at
    the same conclusion: closing that gap needs a purpose-built leaf/non-leaf classifier
    trained on real photographs, not a different threshold. See
    audit/test_prototype_rebuild.py and audit/validate_expanded.py.

    reject_threshold is 0.05 because the weakest genuine leaf measured (0.0793) and the
    weakest non-leaf image (0.0808) overlap. Raising it would let the gate tell a real
    leaf that it is not a plant, which is a worse failure than asking for a better photo.

Three states
------------
    VALID_LEAF   leaf_confidence >= accept_threshold
    NON_LEAF     leaf_confidence <  reject_threshold
    UNCERTAIN    in between, and answered with "please provide a clearer image"

The uncertain band is the point of the design: when the evidence is weak the correct
answer is to ask for a better photo, not to pick the least-bad of 38 diseases.

Dependencies: numpy and Pillow only. No torch import, so the gate can be unit-tested
and reasoned about without the model. The caller passes in the embedding and the
posterior it already computed for classification, so the gate adds no forward pass,
and `prefilter` can reject most non-leaves before any forward pass happens at all.
"""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = ROOT / 'artifacts' / 'leaf_gate' / 'leaf_gate_config.json'

ERROR_NON_LEAF = 'NON_LEAF_IMAGE'
ERROR_LEAF_UNCERTAIN = 'LEAF_UNCERTAIN'
ERROR_TOO_SMALL = 'IMAGE_TOO_SMALL'
ERROR_TOO_DARK = 'IMAGE_TOO_DARK'
ERROR_OVEREXPOSED = 'IMAGE_OVEREXPOSED'
ERROR_TOO_BLANK = 'IMAGE_TOO_BLANK'
ERROR_TOO_BLURRY = 'IMAGE_TOO_BLURRY'
ERROR_UNREADABLE = 'IMAGE_UNREADABLE'

MESSAGES = {
    ERROR_NON_LEAF: 'Please upload or capture a clear plant leaf image only.',
    ERROR_LEAF_UNCERTAIN: ("We couldn't confidently identify a plant leaf in this image. "
                           'Please upload or capture a clearer close-up image of a single leaf.'),
    ERROR_TOO_SMALL: 'This image is too small to analyse. Please upload a larger leaf photo.',
    ERROR_TOO_DARK: 'This image is too dark to analyse. Please capture the leaf in better lighting.',
    ERROR_OVEREXPOSED: 'This image is overexposed. Please capture the leaf with less direct light.',
    ERROR_TOO_BLANK: 'This image looks blank or has no visible detail. '
                     'Please capture the leaf clearly.',
    ERROR_TOO_BLURRY: 'This image is too blurry to analyse. Please capture the leaf again in better focus.',
    ERROR_UNREADABLE: 'This file could not be read as an image. Please upload a JPG, PNG or WebP.',
}

CAMERA_MESSAGES = {
    ERROR_NON_LEAF: 'Please capture a clear plant leaf image only.',
    ERROR_LEAF_UNCERTAIN: ("We couldn't confidently identify a plant leaf. "
                           'Please capture a clearer close-up image of a single leaf.'),
    ERROR_TOO_SMALL: 'That frame is too small to analyse. Please move closer to the leaf.',
    ERROR_TOO_DARK: 'That frame is too dark. Please point the camera at a better-lit leaf.',
    ERROR_OVEREXPOSED: 'That frame is overexposed. Please avoid direct light on the leaf.',
    ERROR_TOO_BLANK: 'Nothing visible in that frame. Please point the camera at a leaf.',
    ERROR_TOO_BLURRY: 'That frame is too blurry. Please hold the camera steady and refocus on the leaf.',
    ERROR_UNREADABLE: 'That frame could not be read. Please try capturing again.',
}

# Analysis resolution. Small enough to stay cheap, large enough that texture and
# gradient statistics are meaningful.
ANALYSIS_SIZE = 256

# Relative importance of the four leaf-likeness terms, chosen from measurements rather
# than intuition; see experiments/leaf_gate/calibration_report.json.
#
# vegetation (solid green structure)  - the only term that separates a leaf from green
#                                       texture such as grass or paint.
# coherence (posterior mass + margin)  - the model-based leaf signal, and what rescues a
#                                       dried brown or all-yellow leaf that scores zero
#                                       vegetation because it carries no green at all.
# detail (absence of flat regions)     - separates photographs from wallpapers, smooth
#                                       gradients and abstract renders.
# feature (leaf prototype manifold)    - measured weight is 0.00. The two worst
#                                       non-leaf photographs sit at cosine distance
#                                       0.140 from the leaf prototypes, closer than
#                                       most real field leaves, so this term rewarded
#                                       them instead of separating them. It is still
#                                       computed and reported for diagnosis and can be
#                                       re-enabled by configuration if a larger,
#                                       broader negative set ever supports it.
DEFAULT_WEIGHTS = {
    'vegetation': 0.35,
    'feature': 0.00,
    'coherence': 0.30,
    'detail': 0.35,
}

# Model-free rejection line, used by `prefilter` before any inference is spent.
#
# It cannot be the main reject threshold. At that stage the class-coherence term is not
# available, so only two terms remain, and an isolated red or brown leaf on a plain
# background scores low on both: it has no green, and the background leaves it with
# little detail. Measured over the calibration set, the lowest prefilter score among
# 457 real leaves is 0.1232 (a red cherry leaf isolated on white), while 24 non-leaf
# photographs reach 0.0000 with a 5th percentile of 0.0003. A line of 0.10 therefore
# sits strictly below every measured leaf while still refusing a quarter of the
# non-leaf set without running the classifier at all.
PREFILTER_REJECT_DEFAULT = 0.10

# Quality thresholds. All are intentionally permissive: they only reject images
# that are unusable for any vision task, not merely unusual-looking leaves.
# Exposure bounds are evaluated on the subject, so a white studio background is
# never mistaken for a blown-out photograph.
QUALITY_DEFAULTS = {
    'min_side_px': 64,
    'min_pixels': 64 * 64,
    'min_sharpness': 8.0,                  # variance of the Laplacian on a 0-255 grayscale
    'min_subject_mean_luma': 0.10,         # genuinely near-black subject
    'max_subject_mean_luma': 0.95,         # genuinely blown-out subject
    'max_subject_clipped_fraction': 0.55,  # subject pixels crushed to black or white
    'max_flat_fraction': 0.92,             # fraction of pixels with no local gradient
}


def _to_float_rgb(image):
    small = image.convert('RGB')
    if max(small.size) > ANALYSIS_SIZE:
        scale = ANALYSIS_SIZE / max(small.size)
        small = small.resize((max(1, int(small.width * scale)),
                              max(1, int(small.height * scale))),
                             Image.BILINEAR)
    return np.asarray(small, dtype=np.float32) / 255.0


def _laplacian(gray):
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.zeros((1, 1), dtype=np.float32)
    core = gray[1:-1, 1:-1]
    return (4.0 * core - gray[:-2, 1:-1] - gray[2:, 1:-1]
            - gray[1:-1, :-2] - gray[1:-1, 2:])


def _gradient_magnitude(gray):
    if gray.shape[0] < 2 or gray.shape[1] < 2:
        return np.zeros((1, 1), dtype=np.float32)
    gx = gray[1:, 1:] - gray[:-1, 1:]   # interior gradients, both axes aligned
    gy = gray[1:, 1:] - gray[1:, :-1]
    return np.sqrt(gx * gx + gy * gy)


def _entropy(values, bins=48):
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    if total == 0:
        return 0.0
    p = hist[hist > 0] / float(total)
    return float(-(p * np.log2(p)).sum())


def _erode_twice(mask):
    """Two 3x3 erosions, padded back to the original shape. Pure numpy, no OpenCV."""
    m = mask
    for _ in range(2):
        if m.shape[0] < 5 or m.shape[1] < 5:
            return np.zeros_like(mask)
        core = (m[1:-1, 1:-1]
                & m[:-2, 1:-1] & m[2:, 1:-1]
                & m[1:-1, :-2] & m[1:-1, 2:]
                & m[:-2, :-2] & m[:-2, 2:]
                & m[2:, :-2] & m[2:, 2:])
        padded = np.zeros_like(mask)
        padded[1:-1, 1:-1] = core
        m = padded
    return m


def _ramp(value, lo_bad, lo_ok, hi_ok, hi_bad):
    """Piecewise-linear 0..1 score: 0 below lo_bad, 1 across [lo_ok, hi_ok], 0 above hi_bad."""
    if value <= lo_bad:
        return 0.0
    if value < lo_ok:
        return (value - lo_bad) / float(lo_ok - lo_bad)
    if value <= hi_ok:
        return 1.0
    if value >= hi_bad:
        return 0.0
    return (hi_bad - value) / float(hi_bad - hi_ok)


def vegetation_score(solid_vegetation_cover):
    """Signal A. Solid, contiguous vegetation: a leaf-like blob, not green texture.

    Measured on real photographs: leaf images have a median solid cover of 0.38-0.40,
    non-leaf images 0.02. This is the term that lets an all-brown or all-yellow dried
    leaf through, because such a leaf simply scores 0 here and is then carried by the
    feature and coherence terms instead of being rejected.
    """
    return _ramp(solid_vegetation_cover, 0.010, 0.060, 1.01, 1.01)


def detail_score(flat_fraction):
    """Signal D. Penalises images with large gradient-free areas.

    Measured: leaf images have a median flat fraction of 0.04-0.21, non-leaf images
    0.61. Abstract renders and smooth wallpapers are almost flat; a leaf photograph
    carries detail everywhere. Deliberately gentle, because an out-of-focus field
    photograph is legitimately flat and must not be refused for it.
    """
    if flat_fraction <= 0.30:
        return 1.0
    if flat_fraction >= 0.85:
        return 0.0
    return (0.85 - flat_fraction) / 0.55


def coherence_score(crop_mass, margin):
    """Signal C. Concentration of the posterior on one crop, and the top-1 margin."""
    mass = _ramp(crop_mass, 0.18, 0.35, 1.01, 1.01)
    gap = _ramp(margin, 0.02, 0.15, 1.01, 1.01)
    return 0.5 * mass + 0.5 * gap


def combine_score(terms, weights):
    """Weighted mean over whichever terms are available."""
    available = {k: v for k, v in terms.items() if v is not None}
    if not available:
        return None
    total_weight = sum(weights.get(k, 0.0) for k in available)
    if total_weight <= 0:
        return None
    return sum(weights.get(k, 0.0) * v for k, v in available.items()) / total_weight


def image_statistics(image):
    """Colour-agnostic quality and surface statistics for one image."""
    rgb = _to_float_rgb(image)
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    width, height = image.size

    lap = _laplacian(gray) * 255.0
    sharpness = float(lap.var()) if lap.size else 0.0

    grad = _gradient_magnitude(gray) * 255.0
    flat_fraction = float((grad < 2.0).mean()) if grad.size else 1.0

    mean_luma = float(gray.mean())
    clipped = (gray <= 0.02) | (gray >= 0.98)
    clipped_fraction = float(clipped.mean())

    # Border colour approximates the background for the usual "leaf on a surface"
    # framing. It is only used to separate foreground from background for a surface
    # texture measure and for exposure; no decision is based on it being green or not.
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    background_luma = float(np.median(border))
    foreground = np.abs(gray - background_luma) > 0.08
    foreground_ratio = float(foreground.mean())

    # Exposure is judged on the subject, not on the whole frame. A leaf photographed
    # on a clean white background is a good photograph, but counting that background as
    # clipped highlights refused it as overexposed, which is a false rejection of a
    # perfectly usable image. When there is no separable subject the whole frame is the
    # subject, so a uniformly washed-out photograph is still caught.
    if foreground_ratio >= 0.02:
        subject_mean_luma = float(gray[foreground].mean())
        subject_clipped_fraction = float(clipped[foreground].mean())
    else:
        subject_mean_luma = mean_luma
        subject_clipped_fraction = clipped_fraction

    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    if foreground.shape == lap.shape:
        foreground_mask = foreground
    elif foreground.shape[0] >= lap.shape[0] + 2 and foreground.shape[1] >= lap.shape[1] + 2:
        foreground_mask = foreground[1:-1, 1:-1]
    else:
        foreground_mask = np.ones(lap.shape, dtype=bool)
    surface_texture = float(lap[foreground_mask].std()) if foreground_mask.any() else 0.0
    colour_variety = _entropy(chroma)

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    vegetation_mask = (g > r * 1.04) & (g > b * 1.04) & (g > 0.12)
    green_fraction = float(vegetation_mask.mean())
    # A leaf is a solid, contiguous piece of tissue. Grass, foliage texture and flat
    # green paint are green but not solid, so the vegetation signal is measured after
    # two erosions: what survives is a leaf-like blob rather than a green texture.
    eroded = _erode_twice(vegetation_mask)
    solid_vegetation_cover = float(eroded.mean())
    vegetation_solidity = float(eroded.sum() / max(int(vegetation_mask.sum()), 1))

    return {
        'width': width,
        'height': height,
        'pixels': width * height,
        'sharpness': round(sharpness, 3),
        'flat_fraction': round(flat_fraction, 4),
        'mean_luma': round(mean_luma, 4),
        'clipped_fraction': round(clipped_fraction, 4),
        'subject_mean_luma': round(subject_mean_luma, 4),
        'subject_clipped_fraction': round(subject_clipped_fraction, 4),
        'foreground_ratio': round(foreground_ratio, 4),
        'surface_texture': round(surface_texture, 3),
        'colour_variety': round(colour_variety, 4),
        'solid_vegetation_cover': round(solid_vegetation_cover, 4),
        'vegetation_solidity': round(vegetation_solidity, 4),
        'green_fraction_diagnostic': round(green_fraction, 4),
    }


def quality_error(stats, rules=None):
    """Return an error code when the image is unusable for any vision task, else None.

    Order matters: exposure is judged before sharpness because a genuinely dark
    photo is also soft, and "too dark" is far more actionable than "too blurry".

    Every exposure bound is evaluated against the subject rather than the full frame.
    Measured on the 457 leaf images used for calibration, subject highlight clipping
    has a maximum of 0.46 and a 95th percentile of 0.04, while whole-frame clipping
    reaches 0.79 on exactly the photographs that must not be refused: leaves shot on
    a white background. Subject luma spans 0.14 to 0.75 over the same set, so these
    bounds do not touch a pale, yellowed or bleached leaf. The 0.55 clipping bound
    still refuses the worst measured non-leaf photographs, which reach 0.75.
    """
    rules = rules or QUALITY_DEFAULTS
    if stats['width'] < rules['min_side_px'] or stats['height'] < rules['min_side_px']:
        return ERROR_TOO_SMALL
    if stats['pixels'] < rules['min_pixels']:
        return ERROR_TOO_SMALL
    if stats['subject_mean_luma'] < rules['min_subject_mean_luma']:
        return ERROR_TOO_DARK
    if stats['subject_mean_luma'] > rules['max_subject_mean_luma']:
        return ERROR_OVEREXPOSED
    if stats['subject_clipped_fraction'] > rules['max_subject_clipped_fraction']:
        return (ERROR_OVEREXPOSED if stats['subject_mean_luma'] >= 0.5
                else ERROR_TOO_DARK)
    if stats['flat_fraction'] > rules['max_flat_fraction']:
        return ERROR_TOO_BLANK
    if stats['sharpness'] < rules['min_sharpness']:
        return ERROR_TOO_BLURRY
    return None


def appearance_score(stats):
    """Green-agnostic plant-surface plausibility in 0..1.

    Retained as a diagnostic only. Measured against real non-leaf photographs it does
    not separate them from leaf photographs at all (0.930 for leaves versus 0.930 for
    non-leaves), so it carries zero weight in the decision.
    """
    foreground = _ramp(stats['foreground_ratio'], 0.03, 0.12, 0.97, 1.01)
    texture = _ramp(stats['surface_texture'], 2.0, 12.0, 400.0, 600.0)
    variety = _ramp(stats['colour_variety'], 1.5, 3.0, 6.5, 7.5)
    parts = {
        'foreground_ratio': round(foreground, 4),
        'surface_texture': round(texture, 4),
        'colour_variety': round(variety, 4),
    }
    score = 0.34 * foreground + 0.33 * texture + 0.33 * variety
    return round(score, 4), parts


def class_coherence(probabilities, crops):
    """Signal C inputs: how much posterior mass sits on the winning crop, and the margin.

    `probabilities` is the 38-way softmax and `crops` the crop prefix of each class.
    """
    import numpy as _np
    probs = _np.asarray(probabilities, dtype=_np.float64).reshape(-1)
    if probs.size == 0:
        return 0.0, 0.0, 0.0
    order = _np.argsort(-probs)
    top1, top2 = int(order[0]), int(order[1]) if probs.size > 1 else int(order[0])
    crop = crops[top1]
    mass = float(sum(p for p, c in zip(probs, crops) if c == crop))
    margin = float(probs[top1] - (probs[top2] if probs.size > 1 else 0.0))
    return crop, mass, margin


class LeafGate:
    """Combines vegetation structure, leaf-manifold proximity and class coherence."""

    def __init__(self, prototypes, accept_threshold, reject_threshold,
                 distance_lo=0.34, distance_hi=0.16, weights=None,
                 neighbours=5, quality_rules=None, source=None,
                 prefilter_reject_threshold=None):
        self.prototypes = np.ascontiguousarray(prototypes, dtype=np.float32)
        norms = np.linalg.norm(self.prototypes, axis=1, keepdims=True)
        self.prototypes = self.prototypes / np.clip(norms, 1e-8, None)
        self.accept_threshold = float(accept_threshold)
        self.reject_threshold = float(reject_threshold)
        self.distance_lo = float(distance_lo)
        self.distance_hi = float(distance_hi)
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.neighbours = int(neighbours)
        self.quality_rules = quality_rules or QUALITY_DEFAULTS
        self.prefilter_reject_threshold = float(
            PREFILTER_REJECT_DEFAULT if prefilter_reject_threshold is None
            else prefilter_reject_threshold)
        self.source = source

    @property
    def enabled(self):
        return self.prototypes.size > 0

    @classmethod
    def load(cls, config_path=None):
        path = Path(config_path or os.getenv('LEAF_GATE_CONFIG') or DEFAULT_CONFIG_PATH)
        if not path.exists():
            return cls(np.zeros((0, 1), dtype=np.float32), 1.0, 1.0, source=None)
        with open(path, encoding='utf-8') as fh:
            cfg = json.load(fh)
        proto_path = Path(cfg.get('prototypes', ''))
        if not proto_path.is_absolute():
            proto_path = path.parent / proto_path
        if not proto_path.exists():
            return cls(np.zeros((0, 1), dtype=np.float32), 1.0, 1.0, source=None)
        data = np.load(proto_path)
        return cls(
            prototypes=data['prototypes'],
            accept_threshold=cfg['accept_threshold'],
            reject_threshold=cfg['reject_threshold'],
            distance_lo=cfg.get('distance_lo', 0.34),
            distance_hi=cfg.get('distance_hi', 0.16),
            weights=cfg.get('weights'),
            neighbours=cfg.get('neighbours', 5),
            quality_rules=cfg.get('quality_rules'),
            prefilter_reject_threshold=cfg.get('prefilter_reject_threshold'),
            source=str(path),
        )

    def feature_score(self, embedding):
        """Mean cosine distance to the k nearest leaf prototypes."""
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return 1.0, 0.0, 0
        vec = vec / norm
        sims = self.prototypes @ vec
        k = min(self.neighbours, sims.shape[0])
        top = np.partition(sims, -k)[-k:]
        return float(np.mean(1.0 - top)), float(top.max()), k

    def feature_likeness(self, distance):
        """Map a prototype distance onto 0..1, where 1 means leaf-like.

        Monotonic in the correct direction: an image *closer* than `distance_hi` is
        maximally leaf-like, and the score falls to zero only once the image is further
        than `distance_lo`. Saturating at 1.0 below the near bound matters, because a
        leaf nearer to the prototype set than any stored example is the strongest
        evidence available and must never be scored as 0.
        """
        span = self.distance_lo - self.distance_hi
        if span <= 1e-6:
            return 1.0 if distance <= self.distance_hi else 0.0
        return float(np.clip((self.distance_lo - distance) / span, 0.0, 1.0))

    def assess(self, image, embedding=None, probabilities=None, crops=None):
        """Full gate decision for a PIL image.

        `embedding` and `probabilities` are the pooled backbone features and the 38-way
        softmax, which the caller already computed for classification, so the gate adds
        no extra model forward pass.
        """
        try:
            image = image.convert('RGB')
        except Exception:
            return self._reject(ERROR_UNREADABLE, None, None)

        stats = image_statistics(image)
        quality = quality_error(stats, self.quality_rules)
        if quality:
            return self._reject(quality, stats, None)

        appear, appear_parts = appearance_score(stats)
        vegetation = vegetation_score(stats['solid_vegetation_cover'])
        detail = detail_score(stats['flat_fraction'])

        feature = feature_parts = None
        if embedding is not None and self.enabled:
            distance, similarity, k = self.feature_score(embedding)
            feature = self.feature_likeness(distance)
            feature_parts = {
                'knn_distance': round(distance, 4),
                'max_similarity': round(similarity, 4),
                'neighbours': k,
            }

        coherence = coherence_parts = None
        if probabilities is not None and crops:
            crop, mass, margin = class_coherence(probabilities, crops)
            coherence = coherence_score(mass, margin)
            coherence_parts = {
                'top_crop': crop,
                'top_crop_mass': round(mass, 4),
                'top1_margin': round(margin, 4),
            }

        combined = combine_score(
            {'vegetation': vegetation, 'feature': feature,
             'coherence': coherence, 'detail': detail},
            self.weights,
        )
        if combined is None:
            return self._reject(ERROR_UNREADABLE, stats, None)

        if combined >= self.accept_threshold:
            state, is_leaf = 'VALID_LEAF', True
        elif combined < self.reject_threshold:
            state, is_leaf = 'NON_LEAF', False
        else:
            state, is_leaf = 'UNCERTAIN', None

        return {
            'state': state,
            'is_leaf': is_leaf,
            'leaf_confidence': round(float(combined), 4),
            'quality_ok': True,
            'error_type': None,
            'reason': {
                'VALID_LEAF': 'Image accepted as a plausible plant leaf.',
                'UNCERTAIN': 'Leaf evidence is weak; not enough to diagnose.',
                'NON_LEAF': 'Image does not look like a plant leaf photograph.',
            }[state],
            'score_basis': 'vegetation+feature+coherence+detail',
            'signals': {
                'vegetation_score': round(vegetation, 4),
                'feature_score': None if feature is None else round(feature, 4),
                'coherence_score': None if coherence is None else round(coherence, 4),
                'detail_score': round(detail, 4),
                'feature_parts': feature_parts,
                'coherence_parts': coherence_parts,
                'appearance_score_diagnostic': appear,
                'appearance_parts_diagnostic': appear_parts,
            },
            'statistics': stats,
        }

    def prefilter(self, image):
        """Model-free rejection, for use before any forward pass is spent.

        Returns a rejection dict when the image can be refused using only the quality
        gate and the image statistics, otherwise None. This exists so that a photo of a
        wall or a face costs no inference: `combine_score` renormalises over whichever
        terms are available, so an image with neither solid vegetation nor photographic
        detail scores near zero here.

        The threshold is deliberately far lower than the main reject threshold, because
        the class-coherence term that normally rescues a leaf is not available yet; see
        PREFILTER_REJECT_DEFAULT for the measurement behind the value.
        """
        if not self.enabled:
            return None
        try:
            image = image.convert('RGB')
        except Exception:
            return self._reject(ERROR_UNREADABLE, None, None)

        stats = image_statistics(image)
        quality = quality_error(stats, self.quality_rules)
        if quality:
            return self._reject(quality, stats, None)

        vegetation = vegetation_score(stats['solid_vegetation_cover'])
        detail = detail_score(stats['flat_fraction'])
        score = combine_score({'vegetation': vegetation, 'detail': detail}, self.weights)
        if score is not None and score < self.prefilter_reject_threshold:
            return {
                'state': 'NON_LEAF',
                'is_leaf': False,
                'leaf_confidence': round(float(score), 4),
                'quality_ok': True,
                'error_type': ERROR_NON_LEAF,
                'reason': 'Image does not look like a plant leaf photograph.',
                'score_basis': 'vegetation+detail',
                'stage': 'prefilter',
                'signals': {
                    'vegetation_score': round(vegetation, 4),
                    'feature_score': None,
                    'coherence_score': None,
                    'detail_score': round(detail, 4),
                    'feature_parts': None,
                    'coherence_parts': None,
                    'appearance_score_diagnostic': None,
                    'appearance_parts_diagnostic': None,
                },
                'statistics': stats,
            }
        return None

    def _reject(self, error_type, stats, probabilities):
        return {
            'state': 'REJECTED',
            'is_leaf': False,
            'leaf_confidence': 0.0,
            'quality_ok': False,
            'error_type': error_type,
            'reason': MESSAGES.get(error_type, MESSAGES[ERROR_NON_LEAF]),
            'score_basis': 'quality',
            'signals': {'vegetation_score': None, 'feature_score': None,
                        'coherence_score': None, 'detail_score': None,
                        'feature_parts': None, 'coherence_parts': None,
                        'appearance_score_diagnostic': None,
                        'appearance_parts_diagnostic': None},
            'statistics': stats,
        }


def message_for(error_type, source='upload'):
    table = CAMERA_MESSAGES if source == 'camera' else MESSAGES
    return table.get(error_type, MESSAGES[ERROR_NON_LEAF])


def decode_image(image_bytes):
    """Decode bytes to a PIL image, or return None when unreadable."""
    import io
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        return image.convert('RGB')
    except (UnidentifiedImageError, OSError, ValueError):
        return None
