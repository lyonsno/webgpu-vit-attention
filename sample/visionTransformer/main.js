// Show an error dialog if there's any uncaught exception or promise rejection.
// This gets set up on all pages that include util.ts.
globalThis.addEventListener('unhandledrejection', (ev) => {
    fail(`unhandled promise rejection, please report a bug!
  https://github.com/lyonsno/webgpu-vit-attention/issues/new\n${ev.reason}`);
});
globalThis.addEventListener('error', (ev) => {
    fail(`uncaught exception, please report a bug!
  https://github.com/lyonsno/webgpu-vit-attention/issues/new\n${ev.error}`);
});
/** Shows an error dialog if getting an adapter wasn't successful. */
function quitIfAdapterNotAvailable(adapter) {
    if (!('gpu' in navigator)) {
        fail('navigator.gpu is not defined - WebGPU not available in this browser');
    }
    if (!adapter) {
        fail("requestAdapter returned null - this sample can't run on this system");
    }
}
function quitIfLimitLessThan(adapter, limit, requiredValue, limits) {
    if (limit in adapter.limits) {
        const limitKey = limit;
        const limitValue = adapter.limits[limitKey];
        if (limitValue < requiredValue) {
            fail(`This sample can't run on this system. ${limit} is ${limitValue}, and this sample requires at least ${requiredValue}.`);
        }
        limits[limit] = requiredValue;
    }
}
function supportsDirectBufferBinding(device) {
    const buffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.UNIFORM,
    });
    const layout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: {} }],
    });
    try {
        device.createBindGroup({
            layout,
            entries: [{ binding: 0, resource: buffer }],
        });
        return true;
    }
    catch {
        return false;
    }
    finally {
        buffer.destroy();
    }
}
function supportsDirectTextureBinding(device) {
    const texture = device.createTexture({
        size: [1],
        usage: GPUTextureUsage.TEXTURE_BINDING,
        format: 'rgba8unorm',
    });
    const layout = device.createBindGroupLayout({
        entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} }],
    });
    try {
        device.createBindGroup({
            layout,
            entries: [{ binding: 0, resource: texture }],
        });
        return true;
    }
    catch {
        return false;
    }
    finally {
        texture.destroy();
    }
}
function supportsDirectTextureAttachments(device) {
    const texture = device.createTexture({
        size: [1],
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        format: 'rgba8unorm',
        sampleCount: 4,
    });
    const resolveTarget = device.createTexture({
        size: [1],
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        format: 'rgba8unorm',
    });
    const depthTexture = device.createTexture({
        size: [1],
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
        format: 'depth16unorm',
        sampleCount: 4,
    });
    const encoder = device.createCommandEncoder();
    try {
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                { view: texture, resolveTarget, loadOp: 'load', storeOp: 'store' },
            ],
            depthStencilAttachment: {
                view: depthTexture,
                depthLoadOp: 'load',
                depthStoreOp: 'store',
            },
        });
        pass.end();
        return true;
    }
    catch (e) {
        console.error(e);
        return false;
    }
    finally {
        encoder.finish();
        texture.destroy();
        resolveTarget.destroy();
    }
}
/**
 * Shows an error dialog if getting a adapter or device wasn't successful,
 * or if/when the device is lost or has an uncaptured error. Also checks
 * for direct buffer binding, direct texture binding, and direct texture attachment binding.
 */
function quitIfWebGPUNotAvailableOrMissingFeatures(adapter, device) {
    if (!device) {
        quitIfAdapterNotAvailable(adapter);
        fail('Unable to get a device for an unknown reason');
        return;
    }
    device.lost.then((reason) => {
        fail(`Device lost ("${reason.reason}"):\n${reason.message}`);
    });
    device.addEventListener('uncapturederror', (ev) => {
        fail(`Uncaptured error:\n${ev.error.message}`);
    });
    if (!supportsDirectBufferBinding(device) ||
        !supportsDirectTextureBinding(device) ||
        !supportsDirectTextureAttachments(device)) {
        fail('Core features of WebGPU are unavailable. Please update your browser to a newer version.');
    }
}
/** Fail by showing a console error, and dialog box if possible. */
const fail = (() => {
    function createErrorOutput() {
        if (typeof document === 'undefined') {
            // Not implemented in workers.
            return {
                show(msg) {
                    console.error(msg);
                },
            };
        }
        const dialogBox = document.createElement('dialog');
        dialogBox.close();
        document.body.append(dialogBox);
        const dialogText = document.createElement('pre');
        dialogText.style.whiteSpace = 'pre-wrap';
        dialogBox.append(dialogText);
        const closeBtn = document.createElement('button');
        closeBtn.textContent = 'OK';
        closeBtn.onclick = () => dialogBox.close();
        dialogBox.append(closeBtn);
        return {
            show(msg) {
                // Don't overwrite the dialog message while it's still open
                // (show the first error, not the most recent error).
                if (!dialogBox.open) {
                    dialogText.textContent = msg;
                    dialogBox.showModal();
                }
            },
        };
    }
    let output;
    return (message) => {
        if (!output)
            output = createErrorOutput();
        output.show(message);
        throw new Error(message);
    };
})();

// Weight loading for DeiT-Tiny
// Binary format (v2):
//   magic (4 bytes): "DEIT"
//   version (u32): 2
//   numTensors (u32)
//   For each tensor:
//     nameLen (u32), name (UTF-8)
//     dtype (u32): 0=fp32, 1=int8
//     ndims (u32), shape (ndims * u32)
//     [if int8: scale (f32)]
//     dataLen (u32), [align to 4], data
//
// Int8 tensors are dequantized to fp32 during loading:
//   float_value = int8_value * scale
async function loadWeights(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load weights: ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    const view = new DataView(buffer);
    let offset = 0;
    const magic = String.fromCharCode(view.getUint8(offset), view.getUint8(offset + 1), view.getUint8(offset + 2), view.getUint8(offset + 3));
    offset += 4;
    if (magic !== 'DEIT') {
        throw new Error(`Invalid weight file magic: ${magic}`);
    }
    const version = view.getUint32(offset, true);
    offset += 4;
    if (version !== 2) {
        throw new Error(`Unsupported weight file version: ${version}`);
    }
    const numTensors = view.getUint32(offset, true);
    offset += 4;
    const tensors = new Map();
    for (let t = 0; t < numTensors; t++) {
        const nameLen = view.getUint32(offset, true);
        offset += 4;
        const nameBytes = new Uint8Array(buffer, offset, nameLen);
        const name = new TextDecoder().decode(nameBytes);
        offset += nameLen;
        const dtype = view.getUint32(offset, true);
        offset += 4;
        const ndims = view.getUint32(offset, true);
        offset += 4;
        const shape = [];
        let numElements = 1;
        for (let d = 0; d < ndims; d++) {
            const dim = view.getUint32(offset, true);
            shape.push(dim);
            numElements *= dim;
            offset += 4;
        }
        let scale = 0;
        if (dtype === 1) {
            scale = view.getFloat32(offset, true);
            offset += 4;
        }
        const dataLen = view.getUint32(offset, true);
        offset += 4;
        const alignedOffset = (offset + 3) & -4;
        let data;
        if (dtype === 1) {
            // Int8: dequantize to fp32
            const int8Data = new Int8Array(buffer, alignedOffset, numElements);
            data = new Float32Array(numElements);
            for (let i = 0; i < numElements; i++) {
                data[i] = int8Data[i] * scale;
            }
        }
        else {
            // FP32: read directly
            data = new Float32Array(buffer.slice(alignedOffset, alignedOffset + dataLen));
        }
        offset = alignedOffset + dataLen;
        tensors.set(name, { name, shape, data });
    }
    return { tensors };
}
function createTensorBuffer(device, tensor, usage = GPUBufferUsage.STORAGE) {
    const buf = device.createBuffer({
        size: tensor.data.byteLength,
        usage: usage | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(buf.getMappedRange()).set(tensor.data);
    buf.unmap();
    return buf;
}
// DeiT-Tiny model configuration.
// These values define the network architecture. Changing them requires
// a matching weight file — the shapes of all weight tensors depend on
// these dimensions.
const DEIT_CONFIG = {
    imgSize: 224,
    patchSize: 16,
    numPatches: 196, // (224/16)^2
    numTokens: 197, // numPatches + 1 (CLS token)
    channels: 3,
    dim: 192,
    numHeads: 3,
    headDim: 64, // dim / numHeads
    mlpHiddenDim: 768, // dim * 4
    numLayers: 12,
    numClasses: 1000,
    scale: 1.0 / Math.sqrt(64), // 1/sqrt(headDim)
};

var patchEmbedWGSL = `// Patch embedding compute shader.
// Takes a 224x224x3 image and produces (197, 192) token embeddings:
// 196 patches (14x14 grid of 16x16 patches) + 1 CLS token.
// Each patch is flattened (16*16*3 = 768) then linearly projected to dim 192.
// The inner loop is a naive dot product (768 iterations per thread) for clarity;
// a production implementation would use tiled shared-memory matmul.

struct Params {
  imgSize: u32,   // 224
  patchSize: u32, // 16
  numPatches: u32, // 196
  channels: u32,   // 3
  dim: u32,        // 192
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> image: array<f32>;      // (224*224*3)
@group(0) @binding(2) var<storage, read> projWeight: array<f32>; // (768, 192) = (patchSize*patchSize*channels, dim)
@group(0) @binding(3) var<storage, read> projBias: array<f32>;   // (192)
@group(0) @binding(4) var<storage, read> clsToken: array<f32>;   // (192)
@group(0) @binding(5) var<storage, read> posEmbed: array<f32>;   // (197, 192)
@group(0) @binding(6) var<storage, read_write> output: array<f32>; // (197, 192)

// Compute patch embeddings + CLS token + position embeddings
@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3u,
) {
  let idx = gid.x;
  let numTokens = params.numPatches + 1u; // 197
  let D = params.dim;
  let totalElements = numTokens * D;

  if (idx >= totalElements) { return; }

  let token = idx / D;
  let d = idx % D;

  var val = 0.0;

  if (token == 0u) {
    // CLS token
    val = clsToken[d];
  } else {
    // Patch embedding: flatten patch pixels, project to dim
    let patchIdx = token - 1u;
    let patchSize = params.patchSize;
    let imgSize = params.imgSize;
    let channels = params.channels;
    let patchesPerRow = imgSize / patchSize; // 14

    let patchRow = patchIdx / patchesPerRow;
    let patchCol = patchIdx % patchesPerRow;
    let startY = patchRow * patchSize;
    let startX = patchCol * patchSize;

    // Linear projection: sum over flattened patch
    val = projBias[d];
    let flatDim = patchSize * patchSize * channels; // 768
    for (var i = 0u; i < flatDim; i++) {
      let c = i % channels;
      let pixelInPatch = i / channels;
      let py = pixelInPatch / patchSize;
      let px = pixelInPatch % patchSize;
      let imgY = startY + py;
      let imgX = startX + px;
      let pixelVal = image[(imgY * imgSize + imgX) * channels + c];
      val += pixelVal * projWeight[i * D + d];
    }
  }

  // Add position embedding
  val += posEmbed[idx];

  output[idx] = val;
}
`;

var layerNormWGSL = `// Layer normalization: stabilizes training and inference by normalizing
// each token's activations to zero mean and unit variance, then applying
// learned scale (gamma) and shift (beta) parameters.
//
// Without normalization, activations can drift to extreme values as they
// pass through many layers, causing numerical instability. Layer norm is
// applied before attention and before the MLP in each transformer block.
//
// Each workgroup processes one token (row). Thread 0 computes mean/variance
// serially (D=192 is small enough), then all threads normalize in parallel.

struct Params {
  N: u32,       // number of rows (tokens)
  D: u32,       // dimension per row
  eps: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read> beta: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

var<workgroup> shared_mean: f32;
var<workgroup> shared_inv_std: f32;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3u,
  @builtin(local_invocation_id) local_id: vec3u,
) {
  let row = wg_id.x;
  let tid = local_id.x;
  let D = params.D;
  let base = row * D;

  // Thread 0 computes mean and variance
  if (tid == 0u) {
    var sum = 0.0;
    var sq_sum = 0.0;
    for (var i = 0u; i < D; i++) {
      let val = input[base + i];
      sum += val;
      sq_sum += val * val;
    }
    let mean = sum / f32(D);
    let variance = sq_sum / f32(D) - mean * mean;
    shared_mean = mean;
    shared_inv_std = 1.0 / sqrt(variance + params.eps);
  }
  workgroupBarrier();

  let mean = shared_mean;
  let inv_std = shared_inv_std;

  // All threads normalize and apply affine transform in parallel
  for (var i = tid; i < D; i += 256u) {
    let val = input[base + i];
    output[base + i] = (val - mean) * inv_std * gamma[i] + beta[i];
  }
}
`;

var mlpWGSL = `// MLP (feed-forward network): the "thinking" part of each transformer block.
//
// After attention gathers information from other tokens, the MLP processes
// each token independently through two linear layers with a GELU activation:
//   hidden = GELU(input @ W1 + b1)    — expand from 192 to 768 dims
//   output = hidden @ W2 + b2          — project back from 768 to 192 dims
//
// The 4x expansion lets the network learn richer intermediate representations.
// Two entry points share the same binding layout: 'linearGelu' (fused first
// layer + activation) and 'linear' (plain second layer).

struct Params {
  numRows: u32,  // number of tokens
  inDim: u32,    // input dimension
  outDim: u32,   // output dimension
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

// Tanh-based GELU approximation. WGSL does not have erf(), so we use the
// standard approximation from Hendrycks & Gimpel (2016). The max error vs
// exact GELU is ~3e-4, which is negligible for inference.
fn gelu(x: f32) -> f32 {
  let c = 0.7978845608; // sqrt(2/pi)
  let inner = c * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + tanh(inner));
}

// Linear + GELU: output = GELU(input @ weight + bias)
@compute @workgroup_size(256)
fn linearGelu(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let numRows = params.numRows;
  let inDim = params.inDim;
  let outDim = params.outDim;

  if (idx >= numRows * outDim) { return; }

  let row = idx / outDim;
  let col = idx % outDim;

  var val = bias[col];
  for (var k = 0u; k < inDim; k++) {
    val += input[row * inDim + k] * weight[k * outDim + col];
  }
  output[idx] = gelu(val);
}

// Plain linear: output = input @ weight + bias
@compute @workgroup_size(256)
fn linear(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let numRows = params.numRows;
  let inDim = params.inDim;
  let outDim = params.outDim;

  if (idx >= numRows * outDim) { return; }

  let row = idx / outDim;
  let col = idx % outDim;

  var val = bias[col];
  for (var k = 0u; k < inDim; k++) {
    val += input[row * inDim + k] * weight[k * outDim + col];
  }
  output[idx] = val;
}
`;

var attnScoresWGSL = `// Attention score computation: how much should query token qi attend to key token ki?
//
// For each attention head, computes the scaled dot product between the query
// vector of token qi and the key vector of token ki:
//   score = (Q[qi] . K[ki]) / sqrt(head_dim)
//
// The scaling by 1/sqrt(head_dim) prevents dot products from growing too large
// with increasing dimension, which would push softmax into saturation.
// Each thread computes one element of the (numHeads, N, N) score tensor.

struct Params {
  N: u32,        // number of tokens (197)
  D: u32,        // model dimension (192)
  numHeads: u32, // number of attention heads (3)
  headDim: u32,  // dimension per head (64)
  scale: f32,    // 1/sqrt(headDim)
  layerIdx: u32, // which layer for attnWeights storage offset
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> qBuf: array<f32>;     // (N, D)
@group(0) @binding(2) var<storage, read> kBuf: array<f32>;     // (N, D)
@group(0) @binding(3) var<storage, read_write> scoreBuf: array<f32>; // (numHeads, N, N)

// Compute raw scores
@compute @workgroup_size(256)
fn computeScores(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let N = params.N;
  let numHeads = params.numHeads;
  let headDim = params.headDim;
  let D = params.D;
  let totalScores = numHeads * N * N;

  if (idx >= totalScores) { return; }

  let head = idx / (N * N);
  let remainder = idx % (N * N);
  let qi = remainder / N;
  let ki = remainder % N;

  let headOffset = head * headDim;

  var dot = 0.0;
  for (var d = 0u; d < headDim; d++) {
    dot += qBuf[qi * D + headOffset + d] * kBuf[ki * D + headOffset + d];
  }

  scoreBuf[idx] = dot * params.scale;
}
`;

var attnSoftmaxWGSL = `// Softmax normalization of attention scores.
//
// Converts raw attention scores into a probability distribution: for each
// query token, the attention weights over all key tokens sum to 1.0.
// Uses the numerically stable form: subtract the row maximum before
// exponentiating to prevent overflow.
//
// The resulting attention weights are also stored to a readback buffer
// for visualization — they show which image patches the model attends to.

struct Params {
  N: u32,
  numHeads: u32,
  layerIdx: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> scoreBuf: array<f32>;    // (numHeads, N, N)
@group(0) @binding(2) var<storage, read_write> attnWeights: array<f32>; // (numLayers, numHeads, N, N)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let N = params.N;
  let numHeads = params.numHeads;
  let totalRows = numHeads * N;

  if (idx >= totalRows) { return; }

  let base = idx * N;

  // Find max for numerical stability
  var m = -1e30;
  for (var i = 0u; i < N; i++) {
    m = max(m, scoreBuf[base + i]);
  }

  // Exp and sum
  var s = 0.0;
  for (var i = 0u; i < N; i++) {
    let e = exp(scoreBuf[base + i] - m);
    scoreBuf[base + i] = e;
    s += e;
  }

  // Normalize and store
  let layerOffset = params.layerIdx * numHeads * N * N;
  for (var i = 0u; i < N; i++) {
    let val = scoreBuf[base + i] / s;
    scoreBuf[base + i] = val;
    attnWeights[layerOffset + idx * N + i] = val;
  }
}
`;

var attnApplyWGSL = `// Apply attention weights to value vectors.
//
// For each token, computes a weighted sum of all value vectors using the
// attention probabilities from softmax. Tokens that received high attention
// scores contribute more to the output. This is how the model "reads from"
// the patches it decided to attend to.
//
// Each thread computes one element of the (N, D) output, where the column
// index maps to a specific head and position within that head.

struct Params {
  N: u32,        // number of tokens (197)
  D: u32,        // model dimension (192)
  numHeads: u32, // number of attention heads (3)
  headDim: u32,  // dimension per head (64)
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> scoreBuf: array<f32>;  // (numHeads, N, N)
@group(0) @binding(2) var<storage, read> vBuf: array<f32>;      // (N, D)
@group(0) @binding(3) var<storage, read_write> output: array<f32>; // (N, D)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let idx = gid.x;
  let N = params.N;
  let D = params.D;
  let numHeads = params.numHeads;
  let headDim = params.headDim;

  if (idx >= N * D) { return; }

  let row = idx / D;
  let col = idx % D;
  let head = col / headDim;
  let d = col % headDim;

  var val = 0.0;
  let scoreBase = head * N * N + row * N;
  for (var j = 0u; j < N; j++) {
    val += scoreBuf[scoreBase + j] * vBuf[j * D + head * headDim + d];
  }
  output[idx] = val;
}
`;

// DeiT-Tiny inference engine
// All compute dispatches use at most 7 storage/uniform bindings per bind group.
const C$2 = DEIT_CONFIG;
function ceilDiv(a, b) {
    return Math.ceil(a / b);
}
// Reusable pool of uniform buffers to avoid per-dispatch GPU memory allocation.
// Buffers are allocated on first use and retained for subsequent inference runs.
class UniformBufferPool {
    device;
    buffers = [];
    index = 0;
    constructor(device) {
        this.device = device;
    }
    // Reset the pool index at the start of each run. No buffers are freed.
    reset() {
        this.index = 0;
    }
    // Get a uniform buffer, writing the given data into it.
    get(data) {
        const size = Math.max(data.byteLength, 16);
        if (this.index >= this.buffers.length) {
            this.buffers.push(this.device.createBuffer({
                size,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
        }
        const buf = this.buffers[this.index++];
        this.device.queue.writeBuffer(buf, 0, new Uint8Array(data));
        return buf;
    }
}
/**
 * VitInference runs a full DeiT-Tiny vision transformer forward pass in
 * WebGPU compute shaders.
 *
 * The forward pass follows the standard ViT architecture:
 *   1. Patch embedding: split image into 14x14 grid of 16x16 patches,
 *      project each to a 192-dim token, prepend a CLS token, add position embeddings.
 *   2. 12 transformer blocks, each containing:
 *      - Layer norm → multi-head self-attention (3 heads, 64 dims each) → residual add
 *      - Layer norm → MLP (192 → 768 with GELU → 768 → 192) → residual add
 *   3. Final layer norm → classify using the CLS token → 1000-class logits
 *
 * Attention weights from each layer/head are stored for visualization,
 * allowing interactive exploration of what image regions the model attends to.
 */
class VitInference {
    device;
    uniformPool;
    // Compute pipelines — each shader uses at most 7 bindings per bind group
    patchEmbedPipeline;
    layerNormPipeline;
    linearPipeline; // plain linear projection
    linearGeluPipeline; // linear + GELU
    attnScoresPipeline;
    attnSoftmaxPipeline;
    attnApplyPipeline;
    residualAddPipeline;
    // Buffers
    imageBuffer;
    tokenBuffer; // (197, 192) main token state
    normBuffer; // (197, 192) output of layer norm
    qBuffer; // (197, 192)
    kBuffer; // (197, 192)
    vBuffer; // (197, 192)
    attnOutBuffer; // (197, 192) attention-weighted output
    projOutBuffer; // (197, 192) after output projection
    scoreBuffer; // (3, 197, 197)
    hiddenBuffer; // (197, 768) MLP hidden
    mlpOutBuffer; // (197, 192) MLP output
    classLogitsBuffer; // (1000)
    attnWeightsBuffer; // (12, 3, 197, 197) all attention weights
    logitsReadbackBuffer;
    attnReadbackBuffer;
    // Weight buffers
    layerWeights = [];
    patchEmbedWeights;
    classHeadWeights;
    finalNormWeights;
    constructor(device) {
        this.device = device;
    }
    async initialize(weights) {
        this.uniformPool = new UniformBufferPool(this.device);
        this.createPipelines();
        this.createBuffers();
        this.uploadWeights(weights);
    }
    createPipelines() {
        const device = this.device;
        const makePipeline = (label, code, entryPoint) => device.createComputePipeline({
            label,
            layout: 'auto',
            compute: {
                module: device.createShaderModule({ label, code }),
                entryPoint,
            },
        });
        // Patch embedding: image pixels → token embeddings
        this.patchEmbedPipeline = makePipeline('patchEmbed', patchEmbedWGSL, 'main');
        // Layer normalization: stabilizes activations between layers
        this.layerNormPipeline = makePipeline('layerNorm', layerNormWGSL, 'main');
        // Linear projections (used for Q/K/V, output projection, MLP, and class head).
        // The MLP shader has two entry points with identical bindings: 'linear' (plain)
        // and 'linearGelu' (with GELU activation fused into the output).
        const mlpModule = device.createShaderModule({
            label: 'mlp',
            code: mlpWGSL,
        });
        this.linearGeluPipeline = device.createComputePipeline({
            label: 'linearGelu',
            layout: 'auto',
            compute: { module: mlpModule, entryPoint: 'linearGelu' },
        });
        this.linearPipeline = device.createComputePipeline({
            label: 'linear',
            layout: 'auto',
            compute: { module: mlpModule, entryPoint: 'linear' },
        });
        // Attention is split into three stages to keep bindings per shader low:
        //   1. Scores: Q·K^T scaled by 1/sqrt(head_dim) — measures query-key similarity
        //   2. Softmax: normalizes scores to attention probabilities
        //   3. Apply: weighted sum of values using attention probabilities
        this.attnScoresPipeline = makePipeline('attnScores', attnScoresWGSL, 'computeScores');
        this.attnSoftmaxPipeline = makePipeline('attnSoftmax', attnSoftmaxWGSL, 'main');
        this.attnApplyPipeline = makePipeline('attnApply', attnApplyWGSL, 'main');
        // Element-wise residual addition: enables gradient flow through deep networks
        const residualModule = device.createShaderModule({
            label: 'residualAdd',
            code: `
        @group(0) @binding(0) var<storage, read_write> dst: array<f32>;
        @group(0) @binding(1) var<storage, read> src: array<f32>;
        @group(0) @binding(2) var<uniform> count: u32;

        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) gid: vec3u) {
          if (gid.x >= count) { return; }
          dst[gid.x] = dst[gid.x] + src[gid.x];
        }
      `,
        });
        this.residualAddPipeline = device.createComputePipeline({
            label: 'residualAdd',
            layout: 'auto',
            compute: { module: residualModule, entryPoint: 'main' },
        });
    }
    createBuffers() {
        const device = this.device;
        const T = C$2.numTokens * C$2.dim * 4; // token buffer size in bytes
        const storage = (label, size, extra = 0) => device.createBuffer({
            label,
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | extra,
        });
        // Input image as normalized float32 RGB
        this.imageBuffer = storage('image', C$2.imgSize * C$2.imgSize * C$2.channels * 4, GPUBufferUsage.COPY_DST);
        // Token embeddings: (197 tokens, 192 dims) — the main state through the network
        this.tokenBuffer = storage('tokens', T, GPUBufferUsage.COPY_DST);
        // Scratch buffer for layer norm output
        this.normBuffer = storage('norm', T);
        // Query, Key, Value projections for attention
        this.qBuffer = storage('Q', T);
        this.kBuffer = storage('K', T);
        this.vBuffer = storage('V', T);
        // Attention-weighted value sum (before output projection)
        this.attnOutBuffer = storage('attnOut', T);
        // Output of attention block (after output projection)
        this.projOutBuffer = storage('projOut', T);
        // Attention scores: (3 heads, 197 queries, 197 keys)
        this.scoreBuffer = storage('attnScores', C$2.numHeads * C$2.numTokens * C$2.numTokens * 4);
        // MLP hidden activations: expanded from 192 to 768 dims
        this.hiddenBuffer = storage('mlpHidden', C$2.numTokens * C$2.mlpHiddenDim * 4);
        // MLP output: projected back from 768 to 192 dims
        this.mlpOutBuffer = storage('mlpOut', T);
        // Classification logits: one score per ImageNet class
        this.classLogitsBuffer = storage('classLogits', C$2.numClasses * 4);
        // Stored attention weights from all layers for visualization readback
        this.attnWeightsBuffer = storage('attnWeights', C$2.numLayers * C$2.numHeads * C$2.numTokens * C$2.numTokens * 4);
        this.logitsReadbackBuffer = device.createBuffer({
            label: 'logitsReadback',
            size: C$2.numClasses * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
        this.attnReadbackBuffer = device.createBuffer({
            label: 'attnReadback',
            size: C$2.numLayers * C$2.numHeads * C$2.numTokens * C$2.numTokens * 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });
    }
    uploadWeights(weights) {
        const device = this.device;
        const t = weights.tensors;
        this.patchEmbedWeights = {
            projWeight: createTensorBuffer(device, t.get('patch_embed.proj.weight')),
            projBias: createTensorBuffer(device, t.get('patch_embed.proj.bias')),
            clsToken: createTensorBuffer(device, t.get('cls_token')),
            posEmbed: createTensorBuffer(device, t.get('pos_embed')),
        };
        for (let l = 0; l < C$2.numLayers; l++) {
            const bufs = new Map();
            const prefix = `blocks.${l}`;
            for (const name of [
                'attn.qkv.weight',
                'attn.qkv.bias',
                'attn.proj.weight',
                'attn.proj.bias',
                'norm1.weight',
                'norm1.bias',
                'norm2.weight',
                'norm2.bias',
                'mlp.fc1.weight',
                'mlp.fc1.bias',
                'mlp.fc2.weight',
                'mlp.fc2.bias',
            ]) {
                const tensor = t.get(`${prefix}.${name}`);
                if (tensor)
                    bufs.set(name, createTensorBuffer(device, tensor));
            }
            this.layerWeights.push(bufs);
        }
        this.finalNormWeights = {
            gamma: createTensorBuffer(device, t.get('norm.weight')),
            beta: createTensorBuffer(device, t.get('norm.bias')),
        };
        this.classHeadWeights = {
            weight: createTensorBuffer(device, t.get('head.weight')),
            bias: createTensorBuffer(device, t.get('head.bias')),
        };
    }
    uploadImage(imageData) {
        this.device.queue.writeBuffer(this.imageBuffer, 0, imageData);
    }
    /**
     * Run the full forward pass: image → classification logits + attention maps.
     *
     * All compute work is recorded into a single command encoder and submitted
     * in one GPU queue submission. The results are read back asynchronously
     * via two mapAsync calls in parallel.
     */
    async run() {
        const device = this.device;
        const startTime = performance.now();
        this.uniformPool.reset();
        const encoder = device.createCommandEncoder();
        // Stage 1: Convert the 224x224 RGB image into 197 token embeddings.
        // The image is split into a 14x14 grid of 16x16-pixel patches, each
        // linearly projected to 192 dimensions. A learnable CLS (classification)
        // token is prepended, and position embeddings are added.
        this.encodePatchEmbed(encoder);
        // Stage 2: Pass tokens through 12 transformer blocks. Each block lets
        // every token attend to every other token (self-attention), then processes
        // each token independently through an MLP. The token representations are
        // progressively refined — early layers detect local features like edges
        // and textures, while later layers capture global semantic relationships.
        for (let l = 0; l < C$2.numLayers; l++) {
            this.encodeTransformerBlock(encoder, l);
        }
        // Stage 3: Final layer norm + classification. The CLS token (index 0)
        // has aggregated information from all image patches via attention. We
        // project it to 1000 dimensions — one logit per ImageNet class.
        this.encodeLayerNorm(encoder, this.tokenBuffer, this.normBuffer, this.finalNormWeights.gamma, this.finalNormWeights.beta);
        this.encodeClassHead(encoder);
        // Copy both logits and attention weights for readback in one submit
        const attnBytes = C$2.numLayers * C$2.numHeads * C$2.numTokens * C$2.numTokens * 4;
        encoder.copyBufferToBuffer(this.classLogitsBuffer, 0, this.logitsReadbackBuffer, 0, C$2.numClasses * 4);
        encoder.copyBufferToBuffer(this.attnWeightsBuffer, 0, this.attnReadbackBuffer, 0, attnBytes);
        device.queue.submit([encoder.finish()]);
        // Read both results after a single submit
        const [logits, attnWeights] = await Promise.all([
            this.logitsReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
                const data = new Float32Array(this.logitsReadbackBuffer.getMappedRange(0, C$2.numClasses * 4).slice(0));
                this.logitsReadbackBuffer.unmap();
                return data;
            }),
            this.attnReadbackBuffer.mapAsync(GPUMapMode.READ).then(() => {
                const data = new Float32Array(this.attnReadbackBuffer.getMappedRange(0, attnBytes).slice(0));
                this.attnReadbackBuffer.unmap();
                return data;
            }),
        ]);
        return { logits, attnWeights, elapsedMs: performance.now() - startTime };
    }
    // --- Dispatch helpers (each uses <= 7 bindings) ---
    encodePatchEmbed(encoder) {
        const device = this.device;
        const params = this.uniformPool.get(new Uint32Array([C$2.imgSize, C$2.patchSize, C$2.numPatches, C$2.channels, C$2.dim])
            .buffer);
        const bg = device.createBindGroup({
            layout: this.patchEmbedPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: params } },
                { binding: 1, resource: { buffer: this.imageBuffer } },
                { binding: 2, resource: { buffer: this.patchEmbedWeights.projWeight } },
                { binding: 3, resource: { buffer: this.patchEmbedWeights.projBias } },
                { binding: 4, resource: { buffer: this.patchEmbedWeights.clsToken } },
                { binding: 5, resource: { buffer: this.patchEmbedWeights.posEmbed } },
                { binding: 6, resource: { buffer: this.tokenBuffer } },
            ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.patchEmbedPipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(ceilDiv(C$2.numTokens * C$2.dim, 256));
        pass.end();
    }
    encodeLayerNorm(encoder, input, output, gamma, beta) {
        const device = this.device;
        const paramsData = new ArrayBuffer(16);
        const v = new DataView(paramsData);
        v.setUint32(0, C$2.numTokens, true);
        v.setUint32(4, C$2.dim, true);
        v.setFloat32(8, 1e-6, true);
        const params = this.uniformPool.get(paramsData);
        const bg = device.createBindGroup({
            layout: this.layerNormPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: params } },
                { binding: 1, resource: { buffer: input } },
                { binding: 2, resource: { buffer: gamma } },
                { binding: 3, resource: { buffer: beta } },
                { binding: 4, resource: { buffer: output } },
            ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.layerNormPipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(C$2.numTokens);
        pass.end();
    }
    encodeLinear(encoder, pipeline, input, weight, bias, output, numRows, inDim, outDim) {
        const device = this.device;
        const params = this.uniformPool.get(new Uint32Array([numRows, inDim, outDim]).buffer);
        const bg = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: params } },
                { binding: 1, resource: { buffer: input } },
                { binding: 2, resource: { buffer: weight } },
                { binding: 3, resource: { buffer: bias } },
                { binding: 4, resource: { buffer: output } },
            ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(ceilDiv(numRows * outDim, 256));
        pass.end();
    }
    encodeResidualAdd(encoder, dst, src) {
        const device = this.device;
        const count = C$2.numTokens * C$2.dim;
        const countBuf = this.uniformPool.get(new Uint32Array([count]).buffer);
        const bg = device.createBindGroup({
            layout: this.residualAddPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: dst } },
                { binding: 1, resource: { buffer: src } },
                { binding: 2, resource: { buffer: countBuf } },
            ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(this.residualAddPipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(ceilDiv(count, 256));
        pass.end();
    }
    /**
     * Encodes one transformer block (pre-norm architecture):
     *   norm1 → attention → residual add → norm2 → MLP → residual add
     *
     * Each block refines the token representations. Early layers learn local
     * features (edges, textures), later layers learn global relationships.
     * The residual connections let information flow directly through the network.
     */
    encodeTransformerBlock(encoder, layerIdx) {
        const lw = this.layerWeights[layerIdx];
        // 1. Layer norm stabilizes activations before attention
        this.encodeLayerNorm(encoder, this.tokenBuffer, this.normBuffer, lw.get('norm1.weight'), lw.get('norm1.bias'));
        // 2. Self-attention: normBuffer -> projOutBuffer
        this.encodeAttention(encoder, layerIdx, lw);
        // 3. Residual: tokenBuffer += projOutBuffer
        this.encodeResidualAdd(encoder, this.tokenBuffer, this.projOutBuffer);
        // 4. LayerNorm2: tokenBuffer -> normBuffer
        this.encodeLayerNorm(encoder, this.tokenBuffer, this.normBuffer, lw.get('norm2.weight'), lw.get('norm2.bias'));
        // 5. MLP: normBuffer -> hidden (GELU) -> mlpOutBuffer
        this.encodeLinear(encoder, this.linearGeluPipeline, this.normBuffer, lw.get('mlp.fc1.weight'), lw.get('mlp.fc1.bias'), this.hiddenBuffer, C$2.numTokens, C$2.dim, C$2.mlpHiddenDim);
        this.encodeLinear(encoder, this.linearPipeline, this.hiddenBuffer, lw.get('mlp.fc2.weight'), lw.get('mlp.fc2.bias'), this.mlpOutBuffer, C$2.numTokens, C$2.mlpHiddenDim, C$2.dim);
        // 6. Residual: tokenBuffer += mlpOutBuffer
        this.encodeResidualAdd(encoder, this.tokenBuffer, this.mlpOutBuffer);
    }
    /**
     * Multi-head self-attention: lets each token "look at" every other token.
     *
     * Each token is projected into Query (what am I looking for?), Key (what do
     * I contain?), and Value (what information do I carry?) vectors. Attention
     * scores = Q·K^T / sqrt(head_dim) measure how relevant each key is to each
     * query. After softmax normalization, these scores weight the values to
     * produce the output. Multiple heads allow attending to different aspects
     * simultaneously (e.g., color, shape, position).
     *
     * The attention weights (softmax output) are stored for visualization —
     * they reveal which image patches the model focuses on at each layer/head.
     */
    encodeAttention(encoder, layerIdx, lw) {
        const device = this.device;
        const qkvWeight = lw.get('attn.qkv.weight');
        const qkvBias = lw.get('attn.qkv.bias');
        const wSize = C$2.dim * C$2.dim * 4;
        const bSize = C$2.dim * 4;
        // Q projection (5 bindings)
        this.encodeLinearWithOffsets(encoder, this.linearPipeline, this.normBuffer, qkvWeight, 0, wSize, qkvBias, 0, bSize, this.qBuffer, C$2.numTokens, C$2.dim, C$2.dim);
        // K projection (5 bindings)
        this.encodeLinearWithOffsets(encoder, this.linearPipeline, this.normBuffer, qkvWeight, wSize, wSize, qkvBias, bSize, bSize, this.kBuffer, C$2.numTokens, C$2.dim, C$2.dim);
        // V projection (5 bindings)
        this.encodeLinearWithOffsets(encoder, this.linearPipeline, this.normBuffer, qkvWeight, 2 * wSize, wSize, qkvBias, 2 * bSize, bSize, this.vBuffer, C$2.numTokens, C$2.dim, C$2.dim);
        // Attention scores: Q, K -> scoreBuf (4 bindings)
        {
            const paramsData = new ArrayBuffer(24);
            const v = new DataView(paramsData);
            v.setUint32(0, C$2.numTokens, true);
            v.setUint32(4, C$2.dim, true);
            v.setUint32(8, C$2.numHeads, true);
            v.setUint32(12, C$2.headDim, true);
            v.setFloat32(16, C$2.scale, true);
            v.setUint32(20, layerIdx, true);
            const params = this.uniformPool.get(paramsData);
            const bg = device.createBindGroup({
                layout: this.attnScoresPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: params } },
                    { binding: 1, resource: { buffer: this.qBuffer } },
                    { binding: 2, resource: { buffer: this.kBuffer } },
                    { binding: 3, resource: { buffer: this.scoreBuffer } },
                ],
            });
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.attnScoresPipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(ceilDiv(C$2.numHeads * C$2.numTokens * C$2.numTokens, 256));
            pass.end();
        }
        // Softmax + store attention weights (3 bindings)
        {
            const paramsData = new ArrayBuffer(16);
            const v = new DataView(paramsData);
            v.setUint32(0, C$2.numTokens, true);
            v.setUint32(4, C$2.numHeads, true);
            v.setUint32(8, layerIdx, true);
            const params = this.uniformPool.get(paramsData);
            const bg = device.createBindGroup({
                layout: this.attnSoftmaxPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: params } },
                    { binding: 1, resource: { buffer: this.scoreBuffer } },
                    { binding: 2, resource: { buffer: this.attnWeightsBuffer } },
                ],
            });
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.attnSoftmaxPipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(ceilDiv(C$2.numHeads * C$2.numTokens, 256));
            pass.end();
        }
        // Apply attention: scores, V -> attnOutBuffer (4 bindings)
        {
            const paramsData = new ArrayBuffer(16);
            const v = new DataView(paramsData);
            v.setUint32(0, C$2.numTokens, true);
            v.setUint32(4, C$2.dim, true);
            v.setUint32(8, C$2.numHeads, true);
            v.setUint32(12, C$2.headDim, true);
            const params = this.uniformPool.get(paramsData);
            const bg = device.createBindGroup({
                layout: this.attnApplyPipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: params } },
                    { binding: 1, resource: { buffer: this.scoreBuffer } },
                    { binding: 2, resource: { buffer: this.vBuffer } },
                    { binding: 3, resource: { buffer: this.attnOutBuffer } },
                ],
            });
            const pass = encoder.beginComputePass();
            pass.setPipeline(this.attnApplyPipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(ceilDiv(C$2.numTokens * C$2.dim, 256));
            pass.end();
        }
        // Output projection: attnOutBuffer @ Wo + bo -> projOutBuffer (5 bindings)
        this.encodeLinear(encoder, this.linearPipeline, this.attnOutBuffer, lw.get('attn.proj.weight'), lw.get('attn.proj.bias'), this.projOutBuffer, C$2.numTokens, C$2.dim, C$2.dim);
    }
    encodeLinearWithOffsets(encoder, pipeline, input, weight, weightOffset, weightSize, bias, biasOffset, biasSize, output, numRows, inDim, outDim) {
        const device = this.device;
        const params = this.uniformPool.get(new Uint32Array([numRows, inDim, outDim]).buffer);
        const bg = device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: params } },
                { binding: 1, resource: { buffer: input } },
                {
                    binding: 2,
                    resource: { buffer: weight, offset: weightOffset, size: weightSize },
                },
                {
                    binding: 3,
                    resource: { buffer: bias, offset: biasOffset, size: biasSize },
                },
                { binding: 4, resource: { buffer: output } },
            ],
        });
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(ceilDiv(numRows * outDim, 256));
        pass.end();
    }
    encodeClassHead(encoder) {
        // Classify using the CLS token (index 0 in the token sequence).
        // normBuffer[0..dim] contains the CLS token after final layer norm.
        this.encodeLinear(encoder, this.linearPipeline, this.normBuffer, this.classHeadWeights.weight, this.classHeadWeights.bias, this.classLogitsBuffer, 1, C$2.dim, C$2.numClasses);
    }
}

var visualizeWGSL = `// Attention map visualization render shader.
//
// Blends the original input image with a viridis-colored heatmap showing
// attention intensities. The attention map is a 14x14 grid (one value per
// image patch) that gets bilinearly upsampled to the full image resolution
// by the texture sampler.
//
// The viridis colormap maps low attention (dark purple/blue) to high
// attention (yellow/green), making it easy to see where the model "looks."

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@vertex
fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  // Full-screen triangle
  var pos = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f( 3.0, -1.0),
    vec2f(-1.0,  3.0),
  );

  var output: VertexOutput;
  output.position = vec4f(pos[vertexIndex], 0.0, 1.0);
  output.uv = (pos[vertexIndex] + 1.0) * 0.5;
  output.uv.y = 1.0 - output.uv.y; // flip Y for image coordinates
  return output;
}

@group(0) @binding(0) var imageTex: texture_2d<f32>;
@group(0) @binding(1) var attnTex: texture_2d<f32>;
@group(0) @binding(2) var linearSampler: sampler;

struct VisParams {
  overlayAlpha: f32,
  showOverlay: u32,
}
@group(0) @binding(3) var<uniform> visParams: VisParams;

// Viridis-like colormap
fn viridis(t: f32) -> vec3f {
  let c0 = vec3f(0.267, 0.004, 0.329);
  let c1 = vec3f(0.282, 0.141, 0.458);
  let c2 = vec3f(0.254, 0.265, 0.530);
  let c3 = vec3f(0.207, 0.372, 0.553);
  let c4 = vec3f(0.164, 0.471, 0.558);
  let c5 = vec3f(0.128, 0.567, 0.551);
  let c6 = vec3f(0.135, 0.659, 0.518);
  let c7 = vec3f(0.267, 0.749, 0.441);
  let c8 = vec3f(0.478, 0.821, 0.318);
  let c9 = vec3f(0.741, 0.873, 0.150);
  let c10 = vec3f(0.993, 0.906, 0.144);

  let s = clamp(t, 0.0, 1.0) * 10.0;
  let i = u32(floor(s));
  let f = fract(s);

  var colors = array<vec3f, 11>(c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10);

  let lo = min(i, 10u);
  let hi = min(i + 1u, 10u);
  return mix(colors[lo], colors[hi], f);
}

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4f {
  let imageColor = textureSample(imageTex, linearSampler, input.uv).rgb;

  if (visParams.showOverlay == 0u) {
    return vec4f(imageColor, 1.0);
  }

  let attnValue = textureSample(attnTex, linearSampler, input.uv).r;
  let heatmapColor = viridis(attnValue);

  let blended = mix(imageColor, heatmapColor, visParams.overlayAlpha);
  return vec4f(blended, 1.0);
}
`;

// Attention map visualization
// Extracts CLS token attention over spatial patches, renders as heatmap overlay
const C$1 = DEIT_CONFIG;
class AttentionVisualizer {
    device;
    context;
    pipeline;
    sampler;
    imageTexture;
    attnTexture;
    visParamsBuffer;
    presentationFormat;
    currentLayer = 0;
    currentHead = 0;
    overlayAlpha = 0.6;
    constructor(device, context, presentationFormat) {
        this.device = device;
        this.context = context;
        this.presentationFormat = presentationFormat;
    }
    initialize() {
        const device = this.device;
        const module = device.createShaderModule({ code: visualizeWGSL });
        this.pipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module, entryPoint: 'vs' },
            fragment: {
                module,
                entryPoint: 'fs',
                targets: [{ format: this.presentationFormat }],
            },
            primitive: { topology: 'triangle-list' },
        });
        this.sampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });
        this.imageTexture = device.createTexture({
            size: [C$1.imgSize, C$1.imgSize],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT,
        });
        // Attention map texture: 14x14 (one per grid cell), bilinear upsampled by the shader
        const gridSize = C$1.imgSize / C$1.patchSize; // 14
        this.attnTexture = device.createTexture({
            size: [gridSize, gridSize],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.visParamsBuffer = device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.updateVisParams();
    }
    // Upload the source image as RGBA8
    uploadImage(rgbaData, width, height) {
        this.device.queue.writeTexture({ texture: this.imageTexture }, rgbaData, { bytesPerRow: width * 4, rowsPerImage: height }, { width, height });
    }
    // Update attention map from inference results
    updateAttentionMap(attnWeights, layer, head) {
        this.currentLayer = layer;
        this.currentHead = head;
        const gridSize = C$1.imgSize / C$1.patchSize; // 14
        const N = C$1.numTokens; // 197
        // Extract CLS token (row 0) attention over spatial tokens (columns 1..196)
        const offset = (layer * C$1.numHeads + head) * N * N;
        const clsAttnRaw = new Float32Array(gridSize * gridSize);
        // Normalize to [0, 1] for visualization
        let maxVal = 0;
        for (let i = 0; i < gridSize * gridSize; i++) {
            clsAttnRaw[i] = attnWeights[offset + i + 1]; // skip CLS-to-CLS attention
            maxVal = Math.max(maxVal, clsAttnRaw[i]);
        }
        if (maxVal > 0) {
            for (let i = 0; i < clsAttnRaw.length; i++) {
                clsAttnRaw[i] /= maxVal;
            }
        }
        // Pack into RGBA8 (attention value in R channel, rest zero)
        const rgba = new Uint8Array(gridSize * gridSize * 4);
        for (let i = 0; i < gridSize * gridSize; i++) {
            rgba[i * 4] = Math.round(clsAttnRaw[i] * 255);
            rgba[i * 4 + 1] = 0;
            rgba[i * 4 + 2] = 0;
            rgba[i * 4 + 3] = 255;
        }
        this.device.queue.writeTexture({ texture: this.attnTexture }, rgba, { bytesPerRow: gridSize * 4, rowsPerImage: gridSize }, { width: gridSize, height: gridSize });
    }
    setOverlayAlpha(alpha) {
        this.overlayAlpha = alpha;
        this.updateVisParams();
    }
    updateVisParams() {
        const data = new ArrayBuffer(8);
        const view = new DataView(data);
        view.setFloat32(0, this.overlayAlpha, true);
        view.setUint32(4, 1, true); // showOverlay always on
        this.device.queue.writeBuffer(this.visParamsBuffer, 0, new Uint8Array(data));
    }
    render() {
        const encoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();
        const bindGroup = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this.imageTexture.createView() },
                { binding: 1, resource: this.attnTexture.createView() },
                { binding: 2, resource: this.sampler },
                { binding: 3, resource: { buffer: this.visParamsBuffer } },
            ],
        });
        const pass = encoder.beginRenderPass({
            colorAttachments: [
                {
                    view: textureView,
                    loadOp: 'clear',
                    storeOp: 'store',
                    clearValue: { r: 0, g: 0, b: 0, a: 1 },
                },
            ],
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.draw(3); // full-screen triangle
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }
}
async function loadImageNetLabels(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to load labels: ${response.status}`);
    }
    return response.json();
}
// Get top-K predictions from logits
function topK(logits, labels, k) {
    // Softmax
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
        maxLogit = Math.max(maxLogit, logits[i]);
    }
    const exps = new Float32Array(logits.length);
    let sum = 0;
    for (let i = 0; i < logits.length; i++) {
        exps[i] = Math.exp(logits[i] - maxLogit);
        sum += exps[i];
    }
    for (let i = 0; i < exps.length; i++) {
        exps[i] /= sum;
    }
    // Find top-K
    const indices = Array.from({ length: logits.length }, (_, i) => i);
    indices.sort((a, b) => exps[b] - exps[a]);
    return indices.slice(0, k).map((i) => ({
        label: labels[i] || `class_${i}`,
        probability: exps[i],
        index: i,
    }));
}

// Vision Transformer (DeiT-Tiny) — WebGPU Compute Inference + Attention Visualization
//
// This sample demonstrates how to run a real neural network (a vision
// transformer) entirely in WebGPU compute shaders. It takes an input image,
// splits it into patches, processes them through 12 transformer layers of
// self-attention and feed-forward networks, and outputs both a classification
// (what's in the image?) and attention maps (what did the model look at?).
//
// The attention visualization is the key output: by selecting different layers
// and heads, you can see how the model's focus shifts from local texture
// features in early layers to global semantic understanding in later layers.
const C = DEIT_CONFIG;
const canvas = document.querySelector('canvas');
const dropZone = document.querySelector('#dropZone');
const layerSlider = document.querySelector('#layerSlider');
const headSlider = document.querySelector('#headSlider');
const alphaSlider = document.querySelector('#alphaSlider');
const layerValue = document.querySelector('#layerValue');
const headValue = document.querySelector('#headValue');
const alphaValue = document.querySelector('#alphaValue');
const resultsDiv = document.querySelector('#results');
const statusDiv = document.querySelector('#status');
const adapter = await navigator.gpu?.requestAdapter({
    featureLevel: 'compatibility',
});
quitIfAdapterNotAvailable(adapter);
const limits = {};
quitIfLimitLessThan(adapter, 'maxComputeWorkgroupSizeX', 256, limits);
quitIfLimitLessThan(adapter, 'maxComputeInvocationsPerWorkgroup', 256, limits);
const device = await adapter.requestDevice({ requiredLimits: limits });
quitIfWebGPUNotAvailableOrMissingFeatures(adapter, device);
canvas.width = C.imgSize * 2; // 448 for retina
canvas.height = C.imgSize * 2;
const context = canvas.getContext('webgpu');
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
context.configure({ device, format: presentationFormat });
// --- Initialize components ---
const inference = new VitInference(device);
const visualizer = new AttentionVisualizer(device, context, presentationFormat);
visualizer.initialize();
// --- Load weights and labels ---
statusDiv.textContent = 'Loading model weights...';
let labels = [];
let currentAttnWeights = null;
let modelReady = false;
try {
    const [weights, loadedLabels] = await Promise.all([
        loadWeights('../../assets/models/deit-tiny-int8.bin'),
        loadImageNetLabels('../../assets/models/imagenet-labels.json'),
    ]);
    labels = loadedLabels;
    await inference.initialize(weights);
    modelReady = true;
    statusDiv.textContent = 'Ready. Drag and drop an image to classify.';
}
catch (e) {
    statusDiv.textContent = `Failed to load: ${e instanceof Error ? e.message : String(e)}`;
    console.error(e);
}
// --- Image preprocessing ---
function preprocessImage(img) {
    // Resize to 224x224
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = C.imgSize;
    tmpCanvas.height = C.imgSize;
    const ctx = tmpCanvas.getContext('2d');
    // Center crop: scale shortest side to 224, then center crop
    const scale = Math.max(C.imgSize / img.width, C.imgSize / img.height);
    const sw = img.width * scale;
    const sh = img.height * scale;
    const sx = (C.imgSize - sw) / 2;
    const sy = (C.imgSize - sh) / 2;
    ctx.drawImage(img, sx, sy, sw, sh);
    const imageData = ctx.getImageData(0, 0, C.imgSize, C.imgSize);
    const rgba = imageData.data;
    // ImageNet normalization: (pixel/255 - mean) / std
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    const normalized = new Float32Array(C.imgSize * C.imgSize * C.channels);
    for (let i = 0; i < C.imgSize * C.imgSize; i++) {
        for (let c = 0; c < 3; c++) {
            normalized[i * 3 + c] =
                (imageData.data[i * 4 + c] / 255.0 - mean[c]) / std[c];
        }
    }
    return { normalized, rgba };
}
// --- Run inference ---
let isRunning = false;
async function classifyImage(img) {
    if (isRunning || !modelReady)
        return;
    isRunning = true;
    try {
        statusDiv.textContent = 'Running inference...';
        const { normalized, rgba } = preprocessImage(img);
        // Upload image for visualization
        visualizer.uploadImage(rgba, C.imgSize, C.imgSize);
        // Upload normalized image for inference
        inference.uploadImage(normalized);
        // Run forward pass
        const { logits, attnWeights, elapsedMs } = await inference.run();
        currentAttnWeights = attnWeights;
        // Display results
        const predictions = topK(logits, labels, 5);
        resultsDiv.innerHTML = predictions
            .map((p) => `<div class="prediction"><span>${p.label}</span><span class="prob">${(p.probability * 100).toFixed(1)}%</span></div>`)
            .join('');
        statusDiv.textContent = `Inference: ${elapsedMs.toFixed(0)}ms`;
        // Update attention visualization
        const layer = parseInt(layerSlider.value);
        const head = parseInt(headSlider.value);
        visualizer.updateAttentionMap(attnWeights, layer, head);
        visualizer.render();
    }
    finally {
        isRunning = false;
    }
}
// --- UI event handlers ---
layerSlider.addEventListener('input', () => {
    const layer = parseInt(layerSlider.value);
    layerValue.textContent = String(layer + 1);
    if (currentAttnWeights) {
        visualizer.updateAttentionMap(currentAttnWeights, layer, parseInt(headSlider.value));
        visualizer.render();
    }
});
headSlider.addEventListener('input', () => {
    const head = parseInt(headSlider.value);
    headValue.textContent = String(head + 1);
    if (currentAttnWeights) {
        visualizer.updateAttentionMap(currentAttnWeights, parseInt(layerSlider.value), head);
        visualizer.render();
    }
});
alphaSlider.addEventListener('input', () => {
    const alpha = parseInt(alphaSlider.value);
    alphaValue.textContent = `${alpha}%`;
    visualizer.setOverlayAlpha(alpha / 100);
    if (currentAttnWeights) {
        visualizer.render();
    }
});
// --- Drag and drop ---
function handleFile(file) {
    if (!file.type.startsWith('image/'))
        return;
    const img = new Image();
    img.onload = () => {
        classifyImage(img);
        URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
}
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer?.files.length) {
        handleFile(e.dataTransfer.files[0]);
    }
});
// Also support click to upload
canvas.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = () => {
        if (input.files?.length) {
            handleFile(input.files[0]);
        }
    };
    input.click();
});
// --- Sample image buttons ---
const sampleImagesDiv = document.querySelector('#sampleImages');
async function loadSampleImage(filename) {
    statusDiv.textContent = `Loading ${filename}...`;
    // Highlight active button
    sampleImagesDiv.querySelectorAll('button').forEach((btn) => {
        btn.classList.toggle('active', btn.getAttribute('data-img') === filename);
    });
    try {
        const response = await fetch(`../../assets/img/vit-samples/${filename}`);
        if (!response.ok)
            throw new Error(`${response.status}`);
        const blob = await response.blob();
        const img = new Image();
        img.onload = () => {
            classifyImage(img);
            URL.revokeObjectURL(img.src);
        };
        img.src = URL.createObjectURL(blob);
    }
    catch (e) {
        statusDiv.textContent = `Failed to load ${filename}: ${e}`;
    }
}
sampleImagesDiv.addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn)
        return;
    const imgFile = btn.getAttribute('data-img');
    if (imgFile)
        loadSampleImage(imgFile);
});
// Load first sample image on start (only if model loaded successfully)
if (modelReady) {
    loadSampleImage('golden-retriever.jpg');
}
//# sourceMappingURL=main.js.map
