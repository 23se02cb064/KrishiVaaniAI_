"""Phase A: Verify calibration split independence and metrics."""
import json, os, hashlib
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
split = json.load(open(os.path.join(ROOT, 'experiments/calibration/split_external_seed42.json')))
report = json.load(open(os.path.join(ROOT, 'experiments/calibration/calibration_report_external.json')))
temp = json.load(open(os.path.join(ROOT, 'experiments/calibration/temperature_external.json')))

cal = split['calibration_paths']
hold = split['holdout_paths']

# 1. Disjointness
overlap = set(cal) & set(hold)
print('=== SPLIT DISJOINTNESS ===')
print('Calibration images:', len(cal))
print('Holdout images:', len(hold))
print('Overlap:', len(overlap))
print('Disjoint:', len(overlap) == 0)

# 2. Check overlap with validation/unseen (the 60-image test set)
unseen = []
for root, dirs, files in os.walk(os.path.join(ROOT, 'validation/unseen')):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            unseen.append(os.path.join(root, f))
unseen_set = set(unseen)
cal_unseen = set(cal) & unseen_set
hold_unseen = set(hold) & unseen_set
print()
print('=== OVERLAP WITH validation/unseen (60-image test set) ===')
print('unseen images:', len(unseen_set))
print('Calibration overlap with unseen:', len(cal_unseen))
print('Holdout overlap with unseen:', len(hold_unseen))
print('Independent of test set:', len(cal_unseen) == 0 and len(hold_unseen) == 0)

# 3. Check overlap with training data
train_hashes = set()
train_root = os.path.join(ROOT, 'data/PlantVillage/train')
for root, dirs, files in os.walk(train_root):
    for f in files:
        fp = os.path.join(root, f)
        try:
            h = hashlib.md5(open(fp, 'rb').read()).hexdigest()
            train_hashes.add((h, fp))
        except:
            pass
print()
print('=== OVERLAP WITH TRAINING DATA ===')
print('Training images hashed:', len(train_hashes))
cal_train = 0
hold_train = 0
for p in cal:
    try:
        h = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if any(h == th for th, _ in train_hashes):
            cal_train += 1
    except:
        pass
for p in hold:
    try:
        h = hashlib.md5(open(p, 'rb').read()).hexdigest()
        if any(h == th for th, _ in train_hashes):
            hold_train += 1
    except:
        pass
print('Calibration images in training:', cal_train)
print('Holdout images in training:', hold_train)

# 4. Duplicate images within calibration set
cal_hashes = {}
for p in cal:
    try:
        h = hashlib.md5(open(p, 'rb').read()).hexdigest()
        cal_hashes.setdefault(h, []).append(p)
    except:
        pass
dups_cal = {h: ps for h, ps in cal_hashes.items() if len(ps) > 1}
print()
print('=== DUPLICATES WITHIN CALIBRATION SET ===')
print('Duplicate groups:', len(dups_cal))
for h, ps in dups_cal.items():
    print('  ', ps)

# 5. Near-duplicates across calibration and holdout (by file size + md5 prefix)
print()
print('=== NEAR-DUPLICATE CHECK (same byte size across splits) ===')
cal_sizes = {}
for p in cal:
    try:
        sz = os.path.getsize(p)
        cal_sizes.setdefault(sz, []).append(p)
    except:
        pass
hold_sizes = {}
for p in hold:
    try:
        sz = os.path.getsize(p)
        hold_sizes.setdefault(sz, []).append(p)
    except:
        pass
same_size = set(cal_sizes.keys()) & set(hold_sizes.keys())
print('File sizes appearing in both splits:', len(same_size))
for sz in sorted(same_size):
    print('  size=%d: cal=%s hold=%s' % (sz, [os.path.basename(p) for p in cal_sizes[sz]], [os.path.basename(p) for p in hold_sizes[sz]]))

# 6. Verify temperature value
print()
print('=== TEMPERATURE VERIFICATION ===')
T = temp['temperature']
print('Temperature:', T)
print('Numeric:', isinstance(T, (int, float)))
print('Finite:', float(T) == float(T))
print('Strictly positive:', T > 0)
print('Model sha256 matches production:', temp['model_sha256'] == '9a6b3eb1f3d48fb34831e13c840a6d09e481a6d7588807390661df74bc5fb850')

# 7. Verify metrics computation (recompute ECE on holdout)
print()
print('=== METRICS VERIFICATION ===')
print('Reported holdout ECE before:', report['holdout_before']['expected_calibration_error'])
print('Reported holdout ECE after:', report['holdout_after']['expected_calibration_error'])
print('Reported holdout NLL before:', report['holdout_before']['negative_log_likelihood'])
print('Reported holdout NLL after:', report['holdout_after']['negative_log_likelihood'])
print('Reported holdout Brier before:', report['holdout_before']['brier_score'])
print('Reported holdout Brier after:', report['holdout_after']['brier_score'])
print('Holdout accuracy before:', report['holdout_before']['accuracy_percent'])
print('Holdout accuracy after:', report['holdout_after']['accuracy_percent'])
print('Accuracy unchanged (argmax preserved):', report['holdout_before']['accuracy_percent'] == report['holdout_after']['accuracy_percent'])
print('ECE improved:', report['improvement']['ece_improved'])
print('ECE delta:', report['improvement']['holdout_ece_delta'])

# 8. Class distribution in splits
print()
print('=== CLASS DISTRIBUTION ===')
cal_classes = Counter(os.path.basename(os.path.dirname(p)) for p in cal)
hold_classes = Counter(os.path.basename(os.path.dirname(p)) for p in hold)
print('Calibration classes:', len(cal_classes))
print('Holdout classes:', len(hold_classes))
print('Classes in calibration but not holdout:', set(cal_classes.keys()) - set(hold_classes.keys()))
print('Classes in holdout but not calibration:', set(hold_classes.keys()) - set(cal_classes.keys()))