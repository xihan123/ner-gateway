//! BERT-style tokenizer for Chinese text

use std::collections::HashMap;
use std::path::Path;

use crate::config::{CLS_TOKEN_ID, SEP_TOKEN_ID, UNK_TOKEN_ID};
use crate::error::Result;

/// Tokenizer with vocabulary mapping
pub struct Tokenizer {
    vocab: HashMap<String, i64>,
}

impl Tokenizer {
    /// Load tokenizer from vocabulary file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())?;

        let mut vocab = HashMap::new();

        for (id, line) in content.lines().enumerate() {
            let id = id as i64;
            let token = line.to_string();
            vocab.insert(token, id);
        }

        tracing::info!("Tokenizer loaded: {} tokens", vocab.len());

        Ok(Self { vocab })
    }

    /// Tokenize text into tokens and IDs
    /// For Chinese text, each character becomes a token
    pub fn tokenize(&self, text: &str) -> (Vec<String>, Vec<i64>) {
        let mut tokens = Vec::new();
        let mut ids = Vec::new();

        for ch in text.chars() {
            let s = ch.to_string();
            tokens.push(s.clone());

            let id = self.vocab.get(&s).copied().unwrap_or(UNK_TOKEN_ID);
            ids.push(id);
        }

        (tokens, ids)
    }
    
    /// Prepare model inputs with special tokens
    pub fn prepare_for_model(&self, text: &str, max_length: usize) -> TokenizedInput {
        let clean_text = clean_html(text);
        let (tokens, mut token_ids) = self.tokenize(&clean_text);
        
        // Truncate if too long (reserve 2 for [CLS] and [SEP])
        let max_content_len = max_length.saturating_sub(2);
        if token_ids.len() > max_content_len {
            token_ids.truncate(max_content_len);
        }
        
        let content_len = token_ids.len();
        
        // Add [CLS] at the beginning and [SEP] at the end
        let mut input_ids = vec![CLS_TOKEN_ID];
        input_ids.extend(token_ids);
        input_ids.push(SEP_TOKEN_ID);
        
        let seq_len = input_ids.len();
        
        // Attention mask: all 1s for real tokens
        let attention_mask = vec![1i64; seq_len];
        
        // Token type IDs: all 0s for single sentence
        let token_type_ids = vec![0i64; seq_len];
        
        TokenizedInput {
            input_ids,
            attention_mask,
            token_type_ids,
            tokens: tokens.into_iter().take(content_len).collect(),
        }
    }
}

/// Tokenized input for model
pub struct TokenizedInput {
    pub input_ids: Vec<i64>,
    pub attention_mask: Vec<i64>,
    pub token_type_ids: Vec<i64>,
    pub tokens: Vec<String>,
}

/// Remove HTML tags from text
fn clean_html(text: &str) -> String {
    let re = regex::Regex::new(r"<[^>]+>").unwrap();
    re.replace_all(text, "").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_clean_html() {
        let input = "<p>Hello <b>World</b></p>";
        assert_eq!(clean_html(input), "Hello World");
    }
}
