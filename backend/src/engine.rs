//! ONNX Runtime NER inference engine

use std::sync::Arc;

use ort::{inputs, session::Session, value::Tensor};

use crate::config::{LABEL_BPER, LABEL_IPER, LABEL_O, MAX_SEQ_LENGTH, NUM_LABELS};
use crate::error::Result;
use crate::tokenizer::Tokenizer;

/// NER inference result
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct InferenceResult {
    pub names: Vec<String>,
    pub confidence: f64,
    pub predictions: Vec<i32>,
    pub tokens: Vec<String>,
}

/// NER Engine wrapper for ONNX Runtime
pub struct NEREngine {
    session: Arc<std::sync::Mutex<Session>>,
    tokenizer: Tokenizer,
}

impl NEREngine {
    /// Create a new NER engine
    pub fn new(model_path: &std::path::Path, vocab_path: &std::path::Path) -> Result<Self> {
        tracing::info!("Loading model from: {:?}", model_path);
        
        // Load tokenizer
        let tokenizer = Tokenizer::from_file(vocab_path)?;
        
        // Create ONNX session
        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;
        
        // Log model info
        tracing::info!("Model inputs:");
        for input in session.inputs() {
            tracing::info!("  - {}", input.name());
        }
        tracing::info!("Model outputs:");
        for output in session.outputs() {
            tracing::info!("  - {}", output.name());
        }
        
        Ok(Self { 
            session: Arc::new(std::sync::Mutex::new(session)), 
            tokenizer 
        })
    }
    
    /// Perform NER prediction on text
    pub fn predict(&self, text: &str) -> Result<InferenceResult> {
        // Tokenize
        let input = self.tokenizer.prepare_for_model(text, MAX_SEQ_LENGTH);
        
        if input.input_ids.is_empty() {
            return Ok(InferenceResult {
                names: vec![],
                confidence: 1.0,
                predictions: vec![],
                tokens: vec![],
            });
        }
        
        let seq_len = input.input_ids.len();
        
        // Create input tensors using tuple format (shape, data)
        let shape = [1usize, seq_len];
        
        let input_ids_tensor = Tensor::from_array((
            shape,
            input.input_ids.clone().into_boxed_slice()
        ))?;
        
        let attention_mask_tensor = Tensor::from_array((
            shape,
            input.attention_mask.clone().into_boxed_slice()
        ))?;
        
        let token_type_ids_tensor = Tensor::from_array((
            shape,
            input.token_type_ids.clone().into_boxed_slice()
        ))?;
        
        // Run inference
        let mut session = self.session.lock().unwrap();
        let outputs = session.run(inputs![
            "input_ids" => &input_ids_tensor,
            "attention_mask" => &attention_mask_tensor,
            "token_type_ids" => &token_type_ids_tensor,
        ])?;
        
        // Extract output tensor
        let output = &outputs[0];
        let output_array = output.try_extract_array::<f32>()?;
        
        // Output shape: [1, seq_len, 3]
        // Process predictions
        let mut predictions = Vec::with_capacity(seq_len - 2);
        let mut confidences = Vec::with_capacity(seq_len - 2);
        
        // Skip [CLS] and [SEP]
        for i in 1..(seq_len - 1) {
            let mut logits = [0.0f32; NUM_LABELS];
            for j in 0..NUM_LABELS {
                logits[j] = output_array[[0, i, j]];
            }
            
            // Softmax
            let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
            
            let (pred_idx, &max_logit_val) = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            
            predictions.push(pred_idx as i32);
            let prob = (max_logit_val - max_logit).exp() / sum_exp;
            confidences.push(prob as f64);
        }
        
        // Extract named entities
        let names = extract_entities(&input.tokens, &predictions);
        
        // Calculate average confidence
        let avg_confidence = if confidences.is_empty() {
            1.0
        } else {
            confidences.iter().sum::<f64>() / confidences.len() as f64
        };
        
        Ok(InferenceResult {
            names,
            confidence: avg_confidence,
            predictions,
            tokens: input.tokens,
        })
    }
}

/// Check if a string is a valid person name (not whitespace, punctuation, or other invalid chars)
fn is_valid_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    
    // Check each character
    for ch in name.chars() {
        // Skip whitespace (space, tab, newline, etc.)
        if ch.is_whitespace() {
            continue;
        }
        
        // Skip common punctuation and symbols
        if ch.is_ascii_punctuation() {
            continue;
        }
        
        // Valid character found
        return true;
    }
    
    // All characters were whitespace or punctuation
    false
}

/// Clean name by removing whitespace and invalid characters from edges
fn clean_name(name: &str) -> String {
    name.chars()
        .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
        .collect()
}

/// Extract person names from tokens based on BIO predictions
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
                // Start of new entity
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
                // Continuation of entity
                if in_entity {
                    current_name.push_str(token);
                } else {
                    // I-PER without B-PER, treat as B-PER
                    current_name.clear();
                    current_name.push_str(token);
                    in_entity = true;
                }
            }
            LABEL_O | _ => {
                // End of entity
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
    
    // Don't forget the last entity
    if in_entity && is_valid_name(&current_name) {
        let cleaned = clean_name(&current_name);
        if is_valid_name(&cleaned) {
            names.push(cleaned);
        }
    }
    
    // Deduplicate
    let mut seen = std::collections::HashSet::new();
    names.into_iter()
        .filter(|n| is_valid_name(n) && seen.insert(n.clone()))
        .collect()
}

/// Thread-safe wrapper for NER engine
#[derive(Clone)]
pub struct NEREngineSync(Arc<NEREngine>);

impl NEREngineSync {
    pub fn new(engine: NEREngine) -> Self {
        Self(Arc::new(engine))
    }
    
    pub fn predict(&self, text: &str) -> Result<InferenceResult> {
        self.0.predict(text)
    }
}