// The worker has its own scope and no direct access to functions/objects of the
// global scope. We import the generated JS file to make `wasm_bindgen`
// available which we need to initialize our Wasm code.
importScripts("./pkg/clip_embedding_wasm.js");

console.log("Initializing worker");

async function fetchArrayBuffer(url) {
  const cacheName = "clip-cache";
  const cache = await caches.open(cacheName);
  const cachedResponse = await cache.match(url);
  if (cachedResponse) {
    const data = await cachedResponse.arrayBuffer();
    return new Uint8Array(data);
  }
  const res = await fetch(url, { cache: "force-cache" });
  cache.put(url, res.clone());
  return new Uint8Array(await res.arrayBuffer());
}

// In the worker, we have a different struct that we want to use as in
// `index.js`.
const { test_model } = wasm_bindgen;

async function init_wasm_in_worker() {
  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
  await wasm_bindgen("./pkg/clip_embedding_wasm_bg.wasm");

  // // Create a new object of the `NumberEval` struct.
  // var num_eval = NumberEval.new();

  // Set callback to handle messages passed to the worker.
  self.onmessage = async (event) => {
    // By using methods of a struct as reaction to messages passed to the
    // worker, we can preserve our state between messages.
    // var worker_result = num_eval.is_even(event.data);
    console.time("test_model");
    const result = await test_model();
    console.timeEnd("test_model");

    console.log(result);

    // Send response back to be handled by callback in main thread.
    self.postMessage(result);
  };
}

init_wasm_in_worker();
