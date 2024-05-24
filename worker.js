// The worker has its own scope and no direct access to functions/objects of the
// global scope. We import the generated JS file to make `wasm_bindgen`
// available which we need to initialize our Wasm code.
importScripts("./pkg/clip_embedding_wasm.js");

console.log("Initializing worker");

// async function fetchArrayBuffer(url) {
//   const cacheName = "clip-cache";
//   const cache = await caches.open(cacheName);
//   const cachedResponse = await cache.match(url);
//   if (cachedResponse) {
//     const data = await cachedResponse.arrayBuffer();
//     return new Uint8Array(data);
//   }
//   const res = await fetch(url, { cache: "force-cache" });
//   cache.put(url, res.clone());
//   return new Uint8Array(await res.arrayBuffer());
// }

async function readFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = function (e) {
      const buffer = new Uint8Array(e.target.result);
      resolve(buffer);
    };
    reader.readAsArrayBuffer(file);
  });
}

// In the worker, we have a different struct that we want to use as in
// `index.js`.
const { to_safetensors_buffer, ModelLocation, ClipModel } = wasm_bindgen;

async function init_wasm_in_worker() {
  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`.
  await wasm_bindgen("./pkg/clip_embedding_wasm_bg.wasm");

  // // Create a new object of the `NumberEval` struct.
  // var num_eval = NumberEval.new();

  const modelLocation = new ModelLocation(
    "https://huggingface.co/openai/clip-vit-base-patch32/resolve/refs%2fpr%2f15",
    "model.safetensors",
  );

  self.postMessage({ t: "loadingModel" });

  const clipModel = await new ClipModel(modelLocation);

  console.log("clip model", clipModel);

  self.postMessage({ t: "modelLoaded" });

  self.onmessage = async (event) => {
    switch (event.data.t) {
      case "fileUpload":
        console.log("file", event.data.file);
        const file = await readFile(event.data.file);

        console.time("image features");
        const features = await clipModel.image_features_from_buffer(file);
        console.timeEnd("image features");
        self.postMessage({
          t: "imageFeatures",
          payload: features,
          file,
          imageFilename: event.data.file.name,
        });
        break;

      case "makeFile":
        console.log("make file payload", event.data.payload);
        const buffer = await to_safetensors_buffer(event.data.payload);
        console.log("file buffer", buffer);
        const blob = new Blob([buffer]);
        const url = URL.createObjectURL(blob);
        self.postMessage({
          t: "embeddingFile",
          payload: url,
          imageFilename: event.data.imageFilename,
        });
        break;
    }
  };
}

init_wasm_in_worker();
