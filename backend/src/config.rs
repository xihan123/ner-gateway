//! Application configuration

use std::path::PathBuf;

/// Model configuration constants
pub const MAX_SEQ_LENGTH: usize = 512;
pub const NUM_LABELS: usize = 3;

/// Special token IDs
pub const UNK_TOKEN_ID: i64 = 100;
pub const CLS_TOKEN_ID: i64 = 101;
pub const SEP_TOKEN_ID: i64 = 102;

/// Label IDs for BIO tagging
pub const LABEL_O: i32 = 0;
pub const LABEL_BPER: i32 = 1;
pub const LABEL_IPER: i32 = 2;

/// Default server port
pub const DEFAULT_PORT: u16 = 8080;

/// Application configuration
#[derive(Debug, Clone)]
pub struct Config {
    pub model_path: PathBuf,
    pub vocab_path: PathBuf,
    pub db_path: PathBuf,
    pub port: u16,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("./models/ner_model_int8.onnx"),
            vocab_path: PathBuf::from("./models/vocab.txt"),
            db_path: PathBuf::from("./ner_reviews.db"),
            port: DEFAULT_PORT,
        }
    }
}

impl Config {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(path) = std::env::var("NER_MODEL_PATH") {
            config.model_path = PathBuf::from(path);
        }
        if let Ok(path) = std::env::var("NER_VOCAB_PATH") {
            config.vocab_path = PathBuf::from(path);
        }
        if let Ok(path) = std::env::var("NER_DB_PATH") {
            config.db_path = PathBuf::from(path);
        }
        if let Ok(port) = std::env::var("NER_PORT") {
            if let Ok(port) = port.parse() {
                config.port = port;
            }
        }

        config
    }
}
