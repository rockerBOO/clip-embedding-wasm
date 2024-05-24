"use strict";
const h = React.createElement;

const isAdvancedUpload = (function () {
  var div = document.createElement("div");
  return (
    ("draggable" in div || ("ondragstart" in div && "ondrop" in div)) &&
    "FormData" in window &&
    "FileReader" in window
  );
})();

if (isAdvancedUpload) {
  document.querySelector("#dropbox").classList.add("has-advanced-upload");
}

const dropbox = document.querySelector("#dropbox");

[
  "drag",
  "dragstart",
  "dragend",
  "dragover",
  "dragenter",
  "dragleave",
  // "drop",
].forEach((evtName) =>
  dropbox.addEventListener(evtName, (e) => {
    e.preventDefault();
    e.stopPropagation();
  }),
);

["dragover", "dragenter"].forEach((evtName) => {
  dropbox.addEventListener(evtName, () => {
    dropbox.classList.add("is-dragover");
  });
});

["dragleave", "dragend", "drop"].forEach((evtName) => {
  dropbox.addEventListener(evtName, () => {
    dropbox.classList.remove("is-dragover");
  });
});

async function run_wasm() {
  // Load the wasm file by awaiting the Promise returned by `wasm_bindgen`
  // `wasm_bindgen` was imported in `index.html`
  await wasm_bindgen();

  console.log("index.js loaded");

  // Run main Wasm entry point
  // This will create a worker from within our Rust code compiled to Wasm
  // startup();

  const worker = new Worker("./worker.js");

  worker.addEventListener("message", (msg) => {
    switch (msg.data.t) {
      case "loadingModel":
        console.log("loading model...");
        console.time("loadingModel");
        break;
      case "modelLoaded":
        console.timeEnd("loadingModel");
        dropbox.querySelector("#box_upload").classList.remove("hidden");
        dropbox.querySelector("#box_upload").classList.add("visible");
        document.querySelector("#file").disabled = false;
        dropbox.querySelector("#model_loading").classList.add("hidden");
        worker.postMessage({ t: "imageFeatures" });
        break;

      case "imageFeatures":
        console.log("image features", msg.data.payload);
        showEmbeddings({
          embeddings: msg.data.payload,
          worker,
          file: msg.data.file,
          imageFilename: msg.data.imageFilename,
        });
        break;

      case "embeddingFile":
        console.log("embeddingFile", msg.data.payload);
        // showFile({ file: msg.data.payload });
        break;
    }
  });

  ["drop"].forEach((evtName) => {
    document.addEventListener(evtName, async (e) => {
      e.preventDefault();
      e.stopPropagation();

      const droppedFiles = e.dataTransfer.files;
      for (let i = 0; i < droppedFiles.length; i++) {
        processFile(droppedFiles.item(i));
      }
    });

    document
      .querySelector("#file")
      .addEventListener("change", async function (e) {
        e.preventDefault();
        e.stopPropagation();

        const files = e.target.files;

        for (let i = 0; i < files.length; i++) {
          processFile(files.item(i), worker);
        }
      });
  });
}

function Embedding({ embedding, file }) {
  const blob = new Blob([file]);
  const src = URL.createObjectURL(blob);
  return h(
    "div",
    {},
    h("div", {}, h("h2", {}, "CLIP image embedding")),
    h(
      "div",
      { className: "embedding-results" },
      h("img", { src, className: "embedding-image" }),
      h("textarea", {
        rows: 10,
        className: "embedding",
        value: embedding.join(","),
      }),
    ),
  );
}

function MakeFile({ embeddings, worker, imageFilename }) {
  return embeddings.map((e, i) =>
    h(
      "button",
      {
        key: i,
        onClick: () => {
          worker.postMessage({ t: "makeFile", payload: e, imageFilename });
          worker.addEventListener("message", (e) => {
            switch (e.data.t) {
              case "embeddingFile":
                const a = document.createElement("a");
                a.href = e.data.payload;
                a.download = `${e.data.imageFilename.replace(/\.[^/.]+$/, "")}.safetensors`;

                var container = document.body;
                container.appendChild(a);
                a.click();

                a.remove();
                break;
            }
          });
        },
      },
      "Download embedding file (safetensors)",
    ),
  );
}

function showEmbeddings({ embeddings, worker, file, imageFilename }) {
  dropbox.classList.remove("box__open");
  dropbox.classList.add("box__closed");
  document.querySelector("#jumbo").classList.remove("jumbo__intro");
  // document.querySelector("#note").classList.add("hidden");
  const root = ReactDOM.createRoot(document.getElementById("results"));
  root.render(
    h(
      "div",
      {},
      embeddings.map((embedding, i) =>
        h(Embedding, { key: i, embedding, file }),
      ),
      h(MakeFile, { embeddings, worker, imageFilename }),
    ),
  );
}

function showFile({ file }) {
  const root = ReactDOM.createRoot(document.getElementById("file_con"));
  root.render(
    h(
      "div",
      {},
      h("a", { href: file, download: "embedding.safetensors" }, "Download"),
    ),
  );
}

function addErrorMessage(message) {
  console.error(message);
}

async function processFile(file, worker) {
  worker.postMessage({ t: "fileUpload", file });
}

run_wasm();
