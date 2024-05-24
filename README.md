# CLIP Embedding WASM

Get CLIP image embedding in the browser.

Proof-of-concept towards loading CLIP model into the browser and getting an embedding from the image.

![Screenshot 2024-05-24 at 15-42-41 CLIP embedding](https://github.com/rockerBOO/clip-embedding-wasm/assets/15027/90030f8b-0f3e-42d3-abae-6094a7bd3fdb)

## Features

- Downloads CLIP model from HuggingFace into your browser cache
- CLIP image embedding for uploaded images
- Runs the model locally in your browser

## Build

```bash
wasm-pack build --target no-modules
```

## Usage

```bash
python -m http.server 8000
```

Then go to the page in your browser.

```
http://localhost:8000
```
