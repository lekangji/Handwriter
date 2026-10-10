const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '..', 'webapp', 'capture_tool', 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const context2d = new Proxy({}, { get: (_, property) => property === 'lineWidth' ? 2 : () => {} });
const elements = new Map();
function element(id) {
    if (!elements.has(id)) elements.set(id, {
        width: id === 'calCanvas' ? 300 : 650, height: 150, style: {}, listeners: {}, value: '',
        getContext: () => context2d,
        getBoundingClientRect: () => ({ left: 0, top: 0, width: id === 'calCanvas' ? 300 : 325, height: 75 }),
        addEventListener(type, handler) { this.listeners[type] = handler; },
        setPointerCapture() {}, setAttribute() {}, append() {}, appendChild() {}, click() {},
    });
    return elements.get(id);
}
const storage = new Map();
let exportedBlob;
const context = vm.createContext({
    console, setTimeout, document: { getElementById: element, createElement: () => element(Symbol()) },
    localStorage: { getItem: (key) => storage.get(key) || null, setItem: (key, value) => storage.set(key, value) },
    FileReader: class { readAsText(file) { this.result = JSON.stringify(file.data); this.onload(); } },
    Blob: class { constructor(parts) { this.text = parts.join(''); } },
    URL: { createObjectURL(blob) { exportedBlob = blob; return 'blob:test'; }, revokeObjectURL() {} },
    assert,
});
vm.runInContext(script + `
    pxPerModelUnit = 4;
    assert.equal(resampleStroke([{x: 0, y: 0}, {x: 10, y: 0}]).length, 6);
    assert.equal(resampleStroke([{x: 1, y: 1}, {x: 1, y: 1}]).length, 1);
    assert.equal(resampleStroke([{x: 1, y: 1}]).length, 1);
    const pointer = { isPrimary: true, button: 0, pointerId: 1, clientX: 20, clientY: 30, timeStamp: 1, preventDefault() {} };
    canvas.listeners.pointerdown(pointer);
    canvas.listeners.pointermove({...pointer, pointerId: 2, clientX: 90});
    assert.equal(strokes[0].length, 1, 'unrelated pointers must not corrupt capture');
    canvas.listeners.pointerup({...pointer, type: 'pointerup', clientX: 30, timeStamp: 2});
    saveSample();
    assert.equal(samples.length, 1);
    assert.equal(samples[0].strokes[0][0].x, 20, 'mobile coordinates retain physical calibration');
    assert.equal(currentStroke, null);
    const oldFile = {files: [{data: {calibration_px_per_model_unit: 2, samples: [{text: 'a', strokes: [[[2, 4], [4, 6]]]}]}}]};
    mergePreviousFile({target: oldFile});
    assert.equal(samples.length, 2);
    assert.equal(samples[1].strokes[0][0].x, 4, 'merged sample must retain its original model scale');
    assert.equal(samples[1].strokes[0].length, 2, 'calibrated legacy merge must preserve recorded sampling');
    assert.equal(samples[1].strokes[0][1].y, 12, 'calibrated legacy merge must preserve the endpoint');
    assert.equal(JSON.parse(localStorage.getItem(DRAFT_KEY)).samples.length, 2);
    mergePreviousFile({target: {files: [{data: {calibration_px_per_model_unit: 0, samples: []}}]}});
    assert.equal(samples.length, 2, 'invalid merge must leave saved samples intact');
    mergePreviousFile({target: {files: [{data: [{text: 'a', points: [
        {x: 0, y: 0, pen_down: true}, {x: 0, y: 50, pen_down: false},
        {x: 20, y: 10, pen_down: false},
    ]}]}]}});
    assert.equal(samples.length, 3, 'legacy captures must merge without recapturing');
    assert.equal(samples[2].strokes.length, 2);
    assert.equal(samples[2].strokes[0].at(-1).y, 100, 'estimated legacy scale retains release endpoint');
    assert.equal(samples[2].strokes[1].length, 1, 'legacy dots survive');
`, context);
vm.runInContext('downloadJSON()', context);
const exported = JSON.parse(exportedBlob.text);
assert.equal(exported.calibration_px_per_model_unit, 4);
assert.equal(exported.samples[1].strokes[0].length, 2, 'download must retain imported sampling');
assert.deepEqual(exported.samples[1].strokes[0], [[4, 8], [8, 12]]);
assert.equal('resample_model_units' in exported, false, 'mixed sampling must not claim uniform spacing');
console.log('Capture checks passed: resampling, dots, pointers, responsive calibration, merge, draft.');
