// Regression test for Disease.jsx confidence crash.
// Reproduces: TypeError: Cannot read properties of null (reading 'confidence')
// when result is null (initial component state).
//
// This replicates the exact confidence/display logic from
// frontend/src/pages/Disease.jsx lines 182-187.

const assert = require('node:assert');

function computeDisplay(result) {
  const hasCalibration = !!result && result.calibration_temperature != null;
  const displayConfidence = hasCalibration
    ? (result.confidence_calibrated ?? result.confidence)
    : (result ? result.confidence : 0);
  const displayIsUncertain = hasCalibration
    ? (result.is_uncertain_calibrated ?? result.is_uncertain)
    : (result ? result.is_uncertain : false);
  const confidencePercent = result ? (displayConfidence * 100).toFixed(1) : 0;
  return { displayConfidence, displayIsUncertain, confidencePercent };
}

// Test 1: null result (initial state) must NOT throw
{
  let threw = false;
  try {
    computeDisplay(null);
  } catch (e) {
    threw = true;
  }
  assert.strictEqual(threw, false, 'computeDisplay(null) must not throw TypeError');
  console.log('PASS: computeDisplay(null) does not throw');
}

// Test 2: null result returns safe defaults
{
  const r = computeDisplay(null);
  assert.strictEqual(r.displayConfidence, 0);
  assert.strictEqual(r.displayIsUncertain, false);
  assert.strictEqual(r.confidencePercent, 0);
  console.log('PASS: computeDisplay(null) returns safe defaults');
}

// Test 3: result without calibration uses raw confidence
{
  const r = computeDisplay({ confidence: 0.87, is_uncertain: false });
  assert.strictEqual(r.displayConfidence, 0.87);
  assert.strictEqual(r.confidencePercent, '87.0');
  console.log('PASS: raw confidence used when no calibration');
}

// Test 4: result with calibration uses calibrated confidence
{
  const r = computeDisplay({
    confidence: 0.87,
    confidence_calibrated: 0.65,
    is_uncertain: false,
    is_uncertain_calibrated: true,
    calibration_temperature: 1.5,
  });
  assert.strictEqual(r.displayConfidence, 0.65);
  assert.strictEqual(r.displayIsUncertain, true);
  assert.strictEqual(r.confidencePercent, '65.0');
  console.log('PASS: calibrated confidence used when calibration present');
}

// Test 5: result with calibration but null calibrated value falls back to raw
{
  const r = computeDisplay({
    confidence: 0.87,
    confidence_calibrated: null,
    is_uncertain: false,
    is_uncertain_calibrated: null,
    calibration_temperature: 1.5,
  });
  assert.strictEqual(r.displayConfidence, 0.87);
  assert.strictEqual(r.displayIsUncertain, false);
  console.log('PASS: null calibrated value falls back to raw confidence');
}

console.log('\nAll regression tests passed.');
