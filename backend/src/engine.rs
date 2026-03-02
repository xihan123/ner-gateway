use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::config::{LABEL_BPER, LABEL_IPER, LABEL_O, MAX_SEQ_LENGTH};
use crate::error::Result;
use crate::tokenizer::Tokenizer;
use ndarray::Array2;
use ort::ep::CUDA;
use ort::{session::builder::GraphOptimizationLevel, session::Session, value::TensorRef};

#[cfg(target_os = "windows")]
fn add_cuda_dll_search_paths() {
    use std::env;

    let cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
        r"C:\Program Files\NVIDIA\CUDNN\v9.19\bin\13.1\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.19\bin\12.9\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.18\bin\13.1\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.18\bin\12.9\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.17\bin\13.1\x64",
        r"C:\Program Files\NVIDIA\CUDNN\v9.16\bin\13.1\x64",
    ];

    let current_path = env::var("PATH").unwrap_or_default();
    let mut new_paths: Vec<String> = Vec::new();

    for path in &cuda_paths {
        if std::path::Path::new(path).exists() {
            if !current_path.to_lowercase().contains(&path.to_lowercase()) {
                new_paths.push(path.to_string());
                tracing::info!("Adding to PATH: {}", path);
            }
        }
    }

    if !new_paths.is_empty() {
        let new_path = new_paths.join(";");
        let updated_path = if current_path.is_empty() {
            new_path
        } else {
            format!("{};{}", new_path, current_path)
        };

        unsafe {
            use std::ffi::OsStr;
            use std::os::windows::ffi::OsStrExt;

            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn SetEnvironmentVariableW(name: *const u16, value: *const u16) -> i32;
            }

            let name: Vec<u16> = OsStr::new("PATH")
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let value: Vec<u16> = OsStr::new(&updated_path)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();

            let result = SetEnvironmentVariableW(name.as_ptr(), value.as_ptr());
            if result != 0 {
                tracing::info!("Updated PATH with {} CUDA/cuDNN directories", new_paths.len());
            } else {
                tracing::warn!("Failed to update PATH environment variable");
            }
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn add_cuda_dll_search_paths() {}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct InferenceResult {
    pub names: Vec<String>,
    pub confidence: f64,
    pub predictions: Vec<i32>,
    pub tokens: Vec<String>,
}

pub struct NEREngine {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    using_gpu: bool,
}

impl NEREngine {
    pub fn new(model_path: &std::path::Path, vocab_path: &std::path::Path) -> Result<Self> {
        add_cuda_dll_search_paths();

        tracing::info!("Loading model from: {:?}", model_path);

        let tokenizer = Tokenizer::from_file(vocab_path)?;

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let disable_cuda = std::env::var("NER_DISABLE_CUDA")
            .map(|v| v.to_lowercase() == "true" || v == "1")
            .unwrap_or(false);

        let (session, using_gpu) = if disable_cuda {
            tracing::info!("CUDA disabled by environment variable NER_DISABLE_CUDA");
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(num_threads)?
                .commit_from_file(model_path)?;
            (session, false)
        } else {
            match Self::try_create_gpu_session(model_path, num_threads) {
                Ok((session, true)) => (session, true),
                _ => {
                    tracing::warn!("GPU not available, falling back to CPU");
                    let session = Session::builder()?
                        .with_optimization_level(GraphOptimizationLevel::Level3)?
                        .with_intra_threads(num_threads)?
                        .commit_from_file(model_path)?;
                    (session, false)
                }
            }
        };

        tracing::info!(
            "Model loaded. Threads: {}, GPU: {}",
            num_threads,
            if using_gpu { "YES" } else { "NO" }
        );
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            using_gpu,
        })
    }

    fn try_create_gpu_session(model_path: &std::path::Path, num_threads: usize) -> Result<(Session, bool)> {
        let builder = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(num_threads)?;

        match builder.with_execution_providers([CUDA::default().build()]) {
            Ok(b) => {
                tracing::info!("CUDA execution provider registered");
                if let Ok(device_id) = ort::ep::get_gpu_device() {
                    tracing::info!("GPU device ID: {}", device_id);
                }
                let session = b.commit_from_file(model_path)?;
                Ok((session, true))
            }
            Err(e) => {
                tracing::debug!("CUDA registration failed: {:?}", e);
                Err(e.into())
            }
        }
    }

    pub fn is_using_gpu(&self) -> bool {
        self.using_gpu
    }

    pub fn predict(&self, text: &str) -> Result<InferenceResult> {
        let total_start = Instant::now();

        let tokenize_start = Instant::now();
        let input = self.tokenizer.prepare_for_model(text, MAX_SEQ_LENGTH);
        let tokenize_time = tokenize_start.elapsed();

        if input.input_ids.is_empty() {
            return Ok(InferenceResult {
                names: vec![],
                confidence: 1.0,
                predictions: vec![],
                tokens: vec![],
            });
        }

        let seq_len = input.input_ids.len();

        let input_ids_array = Array2::from_shape_vec((1, seq_len), input.input_ids).unwrap();
        let attention_mask_array = Array2::from_shape_vec((1, seq_len), input.attention_mask).unwrap();
        let token_type_ids_array = Array2::from_shape_vec((1, seq_len), input.token_type_ids).unwrap();

        let input_ids_tensor = TensorRef::from_array_view(input_ids_array.view())?;
        let attention_mask_tensor = TensorRef::from_array_view(attention_mask_array.view())?;
        let token_type_ids_tensor = TensorRef::from_array_view(token_type_ids_array.view())?;

        let mut session = self.session.lock().unwrap();

        let inference_start = Instant::now();
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])?;
        let inference_time = inference_start.elapsed();

        let logits_3d = outputs["logits"]
            .try_extract_array::<f32>()?
            .into_dimensionality::<ndarray::Ix3>()
            .unwrap();

        let mut predictions = Vec::with_capacity(seq_len - 2);
        let mut sum_confidence = 0.0;
        let mut valid_tokens_count = 0;

        for i in 1..(seq_len - 1) {
            let token_logits = logits_3d.slice(ndarray::s![0, i, ..]);

            let mut max_logit = f32::NEG_INFINITY;
            let mut pred_idx = 0;

            for (idx, &val) in token_logits.iter().enumerate() {
                if val > max_logit {
                    max_logit = val;
                    pred_idx = idx;
                }
            }

            predictions.push(pred_idx as i32);

            if pred_idx != LABEL_O as usize {
                let sum_exp: f32 = token_logits.iter().map(|&x| (x - max_logit).exp()).sum();
                let prob = 1.0 / sum_exp;
                sum_confidence += prob as f64;
                valid_tokens_count += 1;
            }
        }

        let names = extract_entities(&input.tokens, &predictions);

        let avg_confidence = if valid_tokens_count == 0 {
            1.0
        } else {
            sum_confidence / valid_tokens_count as f64
        };

        let total_time = total_start.elapsed();

        if !names.is_empty() || tracing::enabled!(tracing::Level::DEBUG) {
            tracing::debug!(
                "Inference: tokenize={:?}, inference={:?}, total={:?}, device={}, seq_len={}, names={:?}",
                tokenize_time,
                inference_time,
                total_time,
                if self.using_gpu { "GPU" } else { "CPU" },
                seq_len,
                names
            );
        }

        Ok(InferenceResult {
            names,
            confidence: avg_confidence,
            predictions,
            tokens: input.tokens,
        })
    }
}

fn is_valid_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    for ch in name.chars() {
        if ch.is_whitespace() {
            continue;
        }
        if ch.is_ascii_punctuation() {
            continue;
        }
        return true;
    }
    false
}

fn clean_name(name: &str) -> String {
    name.chars()
        .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
        .collect()
}

fn extract_entities(tokens: &[String], predictions: &[i32]) -> Vec<String> {
    let mut names = Vec::new();
    let mut current_name = String::new();
    let mut in_entity = false;

    for (i, &pred) in predictions.iter().enumerate() {
        if i >= tokens.len() {
            break;
        }

        let token = &tokens[i];

        match pred {
            LABEL_BPER => {
                if in_entity && is_valid_name(&current_name) {
                    let cleaned = clean_name(&current_name);
                    if is_valid_name(&cleaned) {
                        names.push(cleaned);
                    }
                }
                current_name.clear();
                current_name.push_str(token);
                in_entity = true;
            }
            LABEL_IPER => {
                if in_entity {
                    current_name.push_str(token);
                } else {
                    current_name.clear();
                    current_name.push_str(token);
                    in_entity = true;
                }
            }
            LABEL_O | _ => {
                if in_entity && is_valid_name(&current_name) {
                    let cleaned = clean_name(&current_name);
                    if is_valid_name(&cleaned) {
                        names.push(cleaned);
                    }
                    current_name.clear();
                }
                in_entity = false;
            }
        }
    }

    if in_entity && is_valid_name(&current_name) {
        let cleaned = clean_name(&current_name);
        if is_valid_name(&cleaned) {
            names.push(cleaned);
        }
    }

    let mut seen = std::collections::HashSet::new();
    names
        .into_iter()
        .filter(|n| is_valid_name(n) && seen.insert(n.clone()))
        .collect()
}

#[derive(Clone)]
pub struct NEREngineSync(Arc<NEREngine>);

impl NEREngineSync {
    pub fn new(engine: NEREngine) -> Self {
        Self(Arc::new(engine))
    }

    pub fn predict(&self, text: &str) -> Result<InferenceResult> {
        self.0.predict(text)
    }

    pub fn is_using_gpu(&self) -> bool {
        self.0.is_using_gpu()
    }
}
