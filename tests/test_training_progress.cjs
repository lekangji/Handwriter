const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '..', 'webapp', 'templates', 'train.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {style: {}, textContent: '', addEventListener() {}});
  return elements.get(id);
}
const context = vm.createContext({document: {getElementById: element}, sessionStorage: {getItem() {return null;}}});
vm.runInContext(script, context);
context.job = {status: 'training', progress: 47.5, epoch: 1, epochs: 2, sample: 152, samples: 152,
  processed_steps: 22583, total_steps: 45166, loss: 0, best_loss: -1.2,
  training_elapsed_s: 20, eta_s: null, steps_per_s: 1129.15, device: 'cuda:0',
  device_name: 'Test GPU', backend: 'torch', base_profile: 'lekangv2', learning_rate: 0.00001,
  loss_history: [-1.2]};
vm.runInContext('renderTrainingProgress(job)', context);
assert.equal(element('progressBar').value, 47.5);
assert.equal(element('liveMetrics').style.display, '');
assert.equal(element('liveLoss').textContent, '0.0000');
assert.equal(element('liveHistory').textContent, '1: -1.2000');
assert.equal(element('liveSteps').textContent, '22583/45166');
assert.match(element('liveTime').textContent, /Estimating/);
assert.match(element('liveBase').textContent, /lekangv2/);
assert.match(element('liveDevice').textContent, /Test GPU.*torch/);
context.job = {status: 'validating', progress: 97};
vm.runInContext('renderTrainingProgress(job)', context);
assert.match(element('progressText').textContent, /Checking generated/);
assert.equal(element('liveHistory').textContent, '1: -1.2000');
console.log('Training UI checks passed: live metrics, zero NLL, ETA, epoch history, validation.');
