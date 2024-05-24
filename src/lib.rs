use std::{collections::HashMap, io::Cursor};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip;

use web_sys::console;

use web_sys::{Request, RequestCache, RequestInit, RequestMode, Response};

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

#[cfg(feature = "console_error_panic_hook")]
extern crate console_error_panic_hook;

fn load_image_from_bytes(image: Vec<u8>, image_size: usize) -> Result<Tensor, EmbeddingError> {
    let cursor = Cursor::new(image);
    let img = image::io::Reader::new(cursor)
        .with_guessed_format()?
        .decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );

    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
    Ok(img)
}

async fn fetch_url(url: &str) -> Result<Response, JsValue> {
    let worker: web_sys::WorkerGlobalScope = js_sys::global().unchecked_into();

    let mut opts = RequestInit::new();
    let opts = opts
        .method("GET")
        .mode(RequestMode::Cors)
        .cache(RequestCache::NoCache);

    let request = Request::new_with_str_and_init(url, opts)?;

    console::log_1(&format!("Fetching {url}").into());
    let resp_value = JsFuture::from(worker.fetch_with_request(&request)).await?;

    assert!(resp_value.is_instance_of::<Response>());
    resp_value.dyn_into()
}

async fn process_response(resp: Response) -> Result<Vec<u8>, EmbeddingError> {
    let data = JsFuture::from(resp.blob()?).await?;
    let blob = web_sys::Blob::from(data);

    let array_buffer = JsFuture::from(blob.array_buffer()).await?;
    let data = js_sys::Uint8Array::new(&array_buffer).to_vec();
    Ok(data)
}

async fn fetch_url_with_cache(url: &str) -> Result<Vec<u8>, JsValue> {
    let worker: web_sys::WorkerGlobalScope = js_sys::global().unchecked_into();

    let mut opts = RequestInit::new();
    let opts = opts
        .method("GET")
        .mode(RequestMode::Cors)
        .cache(RequestCache::NoCache);

    let request = Request::new_with_str_and_init(url, opts)?;

    let cache = web_sys::Cache::from(JsFuture::from(worker.caches()?.open("clip-cache")).await?);

    let cache_response = match JsFuture::from(cache.match_with_request(&request)).await {
        Ok(resp) => (!resp.is_undefined()).then(|| web_sys::Response::from(resp)),
        Err(e) => {
            console::log_1(&format!("Error getting cache {e:?}").into());
            None
        }
    };

    if let Some(resp) = cache_response {
        Ok(process_response(resp).await?)
    } else {
        let resp = fetch_url(url).await?;
        JsFuture::from(cache.put_with_request(&request, &resp.clone()?)).await?;
        Ok(process_response(resp).await?)
    }
}

pub enum EmbeddingError {
    Candle(candle_core::Error),
    SafeTensorError(safetensors::SafeTensorError),
    Io(std::io::Error),
    Msg(String),
    Box(Box<dyn std::error::Error + Send + Sync>),
    Image(image::ImageError),
    SerdeWasmBindgen(serde_wasm_bindgen::Error),
    Js(JsValue),
}

impl From<candle_core::Error> for EmbeddingError {
    fn from(value: candle_core::Error) -> Self {
        EmbeddingError::Candle(value)
    }
}

impl From<std::io::Error> for EmbeddingError {
    fn from(value: std::io::Error) -> Self {
        EmbeddingError::Io(value)
    }
}

impl From<JsValue> for EmbeddingError {
    fn from(value: JsValue) -> Self {
        EmbeddingError::Js(value)
    }
}

impl From<Box<dyn std::error::Error + Send + Sync>> for EmbeddingError {
    fn from(value: Box<dyn std::error::Error + Send + Sync>) -> Self {
        EmbeddingError::Box(value)
    }
}

impl From<image::ImageError> for EmbeddingError {
    fn from(value: image::ImageError) -> Self {
        EmbeddingError::Image(value)
    }
}

impl From<serde_wasm_bindgen::Error> for EmbeddingError {
    fn from(value: serde_wasm_bindgen::Error) -> Self {
        EmbeddingError::SerdeWasmBindgen(value)
    }
}

impl From<safetensors::SafeTensorError> for EmbeddingError {
    fn from(value: safetensors::SafeTensorError) -> Self {
        EmbeddingError::SafeTensorError(value)
    }
}

impl From<EmbeddingError> for JsValue {
    fn from(val: EmbeddingError) -> Self {
        match val {
            EmbeddingError::Candle(e) => format!("{e:?}").into(),
            EmbeddingError::Io(e) => format!("{e:?}").into(),
            EmbeddingError::Msg(e) => format!("{e:?}").into(),
            EmbeddingError::Box(e) => format!("{e:?}").into(),
            EmbeddingError::Image(e) => format!("{e:?}").into(),
            EmbeddingError::Js(v) => v,
            EmbeddingError::SerdeWasmBindgen(e) => format!("{e:?}").into(),
            EmbeddingError::SafeTensorError(e) => format!("{e:?}").into(),
        }
    }
}

#[wasm_bindgen]
pub struct ModelLocation {
    base: String,
    model_file: String,
    // tokenizer_file: String,
}

#[wasm_bindgen]
impl ModelLocation {
    #[wasm_bindgen(constructor)]
    pub fn new(base: String, model_file: String) -> ModelLocation {
        ModelLocation { base, model_file }
    }
}

#[wasm_bindgen]
pub struct ClipModel {
    model: clip::ClipModel,
    config: clip::ClipConfig,
    // tokenizer: tokenizers::Tokenizer,
}

#[wasm_bindgen]
impl ClipModel {
    #[wasm_bindgen(constructor)]
    pub async fn from_url(location: ModelLocation) -> Result<ClipModel, EmbeddingError> {
        let device = &Device::Cpu;

        let model_buffer =
            fetch_url_with_cache(&format!("{}/{}", location.base, location.model_file)).await?;
        let vb = VarBuilder::from_buffered_safetensors(model_buffer, DType::F32, device)?;

        let config = clip::ClipConfig::vit_base_patch32();
        let model = clip::ClipModel::new(vb, &config)?;

        // let bytes = fetch_url(&format!("{}/{}", location.base, location.tokenizer_file)).await?;
        //
        // let tokenizer = Tokenizer::from_bytes(bytes)?;

        Ok(ClipModel {
            model,
            config,
            // tokenizer,
        })
    }

    #[wasm_bindgen]
    pub async fn image_features_from_buffer(
        &self,
        buffer: &[u8],
    ) -> Result<JsValue, EmbeddingError> {
        let image = load_image_from_bytes(buffer.to_vec(), self.config.image_size)?;
        let image_features = self
            .model
            .get_image_features(&Tensor::stack(&[image], 0)?)?;
        Ok(serde_wasm_bindgen::to_value(
            &image_features.to_vec2::<f32>()?,
        )?)
    }
}

#[wasm_bindgen]
pub async fn to_safetensors_buffer(vector: Vec<f32>) -> Result<js_sys::Uint8Array, EmbeddingError> {
    let mut tensors = HashMap::new();

    tensors.insert("embedding".to_string(), Tensor::new(vector, &Device::Cpu)?);
    let result: Vec<u8> = safetensors::tensor::serialize(tensors.iter(), &None)?;
    let data = js_sys::Uint8Array::from(&result[..]);
    Ok(data)
}
