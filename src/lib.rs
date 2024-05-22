// use anyhow::Error as E;

use std::io::Cursor;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::clip;

use tokenizers::Tokenizer;
use web_sys::console;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

#[cfg(feature = "console_error_panic_hook")]
extern crate console_error_panic_hook;

// #[derive(Parser)]
// struct Args {
//     #[arg(long)]
//     model: Option<String>,
//
//     #[arg(long)]
//     tokenizer: Option<String>,
//
//     #[arg(long, use_value_delimiter = true)]
//     images: Option<Vec<String>>,
//
//     #[arg(long)]
//     cpu: bool,
//
//     #[arg(long, use_value_delimiter = true)]
//     sequences: Option<Vec<String>>,
// }

async fn load_image<T: AsRef<std::path::Path> + std::fmt::Debug>(
    path: T,
    image_size: usize,
) -> Result<Tensor, EmbeddingError> {
    console::log_1(&format!("load image {path:?}").into());
    let data = fetch_url("https://davelage.com/img/p8.png").await?;
    let cursor = Cursor::new(data);
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
    // .unsqueeze(0)?;
    Ok(img)
}

async fn load_images<T: AsRef<std::path::Path> + std::fmt::Debug>(
    paths: &Vec<T>,
    image_size: usize,
) -> Result<Tensor, EmbeddingError> {
    let mut images = vec![];

    for path in paths {
        let tensor = load_image(path, image_size).await?;
        images.push(tensor);
    }

    let images = Tensor::stack(&images, 0)?;

    Ok(images)
}

#[wasm_bindgen]
extern "C" {
    type Global;

    #[wasm_bindgen(method, getter, js_name = DedicatedWorkerGlobalScope)]
    fn worker(this: &Global) -> JsValue;
}

async fn fetch_url(url: &str) -> Result<Vec<u8>, JsValue> {
    use web_sys::{Request, RequestCache, RequestInit, RequestMode, Response};
    let worker: web_sys::WorkerGlobalScope = js_sys::global().unchecked_into();

    console::log_1(&format!("{worker:?}").into());

    let mut opts = RequestInit::new();
    let opts = opts
        .method("GET")
        .mode(RequestMode::Cors)
        .cache(RequestCache::NoCache);

    let cache = web_sys::Cache::from(JsFuture::from(worker.caches()?.open("clip-cache")).await?);

    let request = Request::new_with_str_and_init(url, opts)?;

    let cache_response = match JsFuture::from(cache.match_with_request(&request)).await {
        Ok(resp) => {
            console::log_1(&format!("Got response: {resp:?}").into());
            match resp.is_undefined() {
                true => None,
                false => Some(web_sys::Response::from(resp)),
            }
        }
        Err(_) => None,
    };

    if let Some(resp) = cache_response {
        console::log_1(&format!("Got cached response: {resp:?}").into());
        let data = JsFuture::from(resp.blob()?).await?;
        let blob = web_sys::Blob::from(data);
        let array_buffer = JsFuture::from(blob.array_buffer()).await?;
        let data = js_sys::Uint8Array::new(&array_buffer).to_vec();
        return Ok(data);
    }

    console::log_1(&format!("Fetching {url}").into());
    let resp_value = JsFuture::from(worker.fetch_with_request(&request)).await?;

    // `resp_value` is a `Response` object.
    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into()?;

    JsFuture::from(cache.put_with_request(&request, &resp.clone()?)).await?;

    let data = JsFuture::from(resp.blob()?).await?;
    let blob = web_sys::Blob::from(data);

    let array_buffer = JsFuture::from(blob.array_buffer()).await?;
    let data = js_sys::Uint8Array::new(&array_buffer).to_vec();
    Ok(data)
}

pub enum EmbeddingError {
    Candle(candle_core::Error),
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
        }
    }
}

pub async fn load_models(
    device: &candle_core::Device,
) -> Result<(clip::ClipModel, clip::ClipConfig, tokenizers::Tokenizer), EmbeddingError> {
    let base_url = "https://huggingface.co/openai/clip-vit-base-patch32/resolve/refs%2Fpr%2F15";
    let model_filename = "model.safetensors";

    let model_buffer = fetch_url(&format!("{base_url}/{model_filename}")).await?;
    let vb = VarBuilder::from_buffered_safetensors(model_buffer, DType::F32, device)?;

    let config = clip::ClipConfig::vit_base_patch32();
    let model = clip::ClipModel::new(vb, &config)?;

    let bytes = fetch_url(
        "https://huggingface.co/openai/clip-vit-base-patch32/raw/refs%2Fpr%2F15/tokenizer.json",
    )
    .await?;

    let tokenizer = tokenizer(bytes)?;

    Ok((model, config, tokenizer))
}

pub async fn inference(
    sequences: Vec<String>,
    imgs: Vec<String>,
    device: &candle_core::Device,
) -> Result<JsValue, EmbeddingError> {
    console_error_panic_hook::set_once();
    let (model, config, tokenizer) = load_models(device).await?;

    // Process inputs

    let (input_ids, vec_seq) = tokenize_sequences(sequences, &tokenizer, device)?;
    let images = load_images(&imgs, config.image_size)
        .await?
        .to_device(device)?;

    let image_features = model.get_image_features(&images)?;

    console::log_2(
        &"image_features:".to_string().into(),
        &serde_wasm_bindgen::to_value(&image_features.to_vec2::<f32>()?)?,
    );

    // let (logits_per_text, logits_per_image) = model.forward(&images, &input_ids)?;

    // console::log_1(&format!("logits per image: {:?}", logits_per_image).into());
    // console::log_1(&format!("logits per text: {:?}", logits_per_text).into());
    //
    // let softmax_image = softmax(&logits_per_image, 1)?;
    //
    // let softmax_image_vec = softmax_image.flatten_all()?.to_vec1::<f32>()?;
    //
    // console::log_1(&format!("softmax_image_vec: {:?}", softmax_image_vec).into());
    //
    // let probability_vec = softmax_image_vec
    //     .iter()
    //     .map(|v| v * 100.0)
    //     .collect::<Vec<f32>>();
    //
    // let probability_per_image = probability_vec.len() / imgs.len();
    //
    // for (i, img) in imgs.iter().enumerate() {
    //     let start = i * probability_per_image;
    //     let end = start + probability_per_image;
    //     let prob = &probability_vec[start..end];
    //     console::log_1(&format!("\n\nResults for image: {}\n", img).into());
    //
    //     for (i, p) in prob.iter().enumerate() {
    //         console::log_1(&format!("Probability: {:.4}% Text: {} ", p, vec_seq[i]).into());
    //     }
    // }

    Ok(serde_wasm_bindgen::to_value(
        &image_features.to_vec2::<f32>()?,
    )?)
    // Ok(())
}

// use std::cell::RefCell;
// use std::rc::Rc;
// use wasm_bindgen::prelude::*;
// use web_sys::{console, HtmlElement, HtmlInputElement, MessageEvent, Worker};
//
//
// /// Run entry point for the main thread.
// #[wasm_bindgen]
// pub fn startup() {
//     // Here, we create our worker. In a larger app, multiple callbacks should be
//     // able to interact with the code in the worker. Therefore, we wrap it in
//     // `Rc<RefCell>` following the interior mutability pattern. Here, it would
//     // not be needed but we include the wrapping anyway as example.
//     let worker_handle = Rc::new(RefCell::new(Worker::new("./worker.js").unwrap()));
//     console::log_1(&"Created a new worker from within Wasm".into());
//
//     // Pass the worker to the function which sets up the `oninput` callback.
//     setup_input_oninput_callback(worker_handle);
// }
//
// fn setup_input_oninput_callback(worker: Rc<RefCell<web_sys::Worker>>) {
//     let document = web_sys::window().unwrap().document().unwrap();
//
//     // If our `onmessage` callback should stay valid after exiting from the
//     // `oninput` closure scope, we need to either forget it (so it is not
//     // destroyed) or store it somewhere. To avoid leaking memory every time we
//     // want to receive a response from the worker, we move a handle into the
//     // `oninput` closure to which we will always attach the last `onmessage`
//     // callback. The initial value will not be used and we silence the warning.
//     #[allow(unused_assignments)]
//     let mut persistent_callback_handle = get_on_msg_callback();
//
//     let callback = Closure::new(move || {
//         console::log_1(&"oninput callback triggered".into());
//         let document = web_sys::window().unwrap().document().unwrap();
//
//         let input_field = document
//             .get_element_by_id("inputNumber")
//             .expect("#inputNumber should exist");
//         let input_field = input_field
//             .dyn_ref::<HtmlInputElement>()
//             .expect("#inputNumber should be a HtmlInputElement");
//
//         // If the value in the field can be parsed to a `i32`, send it to the
//         // worker. Otherwise clear the result field.
//         match input_field.value().parse::<i32>() {
//             Ok(number) => {
//                 // Access worker behind shared handle, following the interior
//                 // mutability pattern.
//                 let worker_handle = &*worker.borrow();
//                 let _ = worker_handle.post_message(&number.into());
//                 persistent_callback_handle = get_on_msg_callback();
//
//                 // Since the worker returns the message asynchronously, we
//                 // attach a callback to be triggered when the worker returns.
//                 worker_handle
//                     .set_onmessage(Some(persistent_callback_handle.as_ref().unchecked_ref()));
//             }
//             Err(_) => {
//                 document
//                     .get_element_by_id("resultField")
//                     .expect("#resultField should exist")
//                     .dyn_ref::<HtmlElement>()
//                     .expect("#resultField should be a HtmlInputElement")
//                     .set_inner_text("");
//             }
//         }
//     });
//
//     // Attach the closure as `oninput` callback to the input field.
//     document
//         .get_element_by_id("inputNumber")
//         .expect("#inputNumber should exist")
//         .dyn_ref::<HtmlInputElement>()
//         .expect("#inputNumber should be a HtmlInputElement")
//         .set_oninput(Some(callback.as_ref().unchecked_ref()));
//
//     // Leaks memory.
//     callback.forget();
// }
//
// /// Create a closure to act on the message returned by the worker
// fn get_on_msg_callback() -> Closure<dyn FnMut(MessageEvent)> {
//     Closure::new(move |event: MessageEvent| {
//         console::log_2(&"Received response: ".into(), &event.data());
//
//         let result = match event.data().as_bool().unwrap() {
//             true => "even",
//             false => "odd",
//         };
//
//         let document = web_sys::window().unwrap().document().unwrap();
//         document
//             .get_element_by_id("resultField")
//             .expect("#resultField should exist")
//             .dyn_ref::<HtmlElement>()
//             .expect("#resultField should be a HtmlInputElement")
//             .set_inner_text(result);
//     })
// }

#[wasm_bindgen]
pub async fn test_model() -> Result<JsValue, JsValue> {
    let sequences = vec!["testing sequences".to_string()];

    // let device = candle_examples::device(args.cpu)?;
    let device = Device::Cpu;

    let vec_imgs = vec![
        "candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg".to_string(),
        "candle-examples/examples/yolo-v8/assets/bike.jpg".to_string(),
    ];

    Ok(inference(sequences, vec_imgs, &device).await?)
}

pub fn tokenizer(bytes: Vec<u8>) -> Result<Tokenizer, EmbeddingError> {
    // {
    //       base_url: "https://huggingface.co/lmz/candle-blip/resolve/main/",
    //       model: "blip-image-captioning-large-q80.gguf",
    //       config: "config.json",
    //       tokenizer: "tokenizer.json",
    //       quantized: true,
    //       size: "505 MB",
    //     }
    // openai/clip-vit-base-patch32

    Ok(Tokenizer::from_bytes(bytes)?)
}

pub fn tokenize_sequences(
    sequences: Vec<String>,
    tokenizer: &Tokenizer,
    device: &Device,
) -> Result<(Tensor, Vec<String>), EmbeddingError> {
    let pad_id = *tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .ok_or(EmbeddingError::Msg("No pad token".to_string()))?;

    let mut tokens = vec![];

    for seq in sequences.clone() {
        let encoding = tokenizer.encode(seq, true)?;
        tokens.push(encoding.get_ids().to_vec());
    }

    let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);

    // Pad the sequences to have the same length
    for token_vec in tokens.iter_mut() {
        let len_diff = max_len - token_vec.len();
        if len_diff > 0 {
            token_vec.extend(vec![pad_id; len_diff]);
        }
    }

    let input_ids = Tensor::new(tokens, device)?;

    Ok((input_ids, sequences))
}
