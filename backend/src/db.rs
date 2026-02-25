//! Database models and repository

use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use rusqlite::{params, types::ToSqlOutput, Connection, Row};
use sha2::{Digest, Sha256};

use crate::error::Result;

/// Review status enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReviewStatus {
    Pending,
    Approved,
    Corrected,
    Rejected,
}

impl ReviewStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Approved => "approved",
            Self::Corrected => "corrected",
            Self::Rejected => "rejected",
        }
    }
    
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "approved" => Some(Self::Approved),
            "corrected" => Some(Self::Corrected),
            "rejected" => Some(Self::Rejected),
            _ => None,
        }
    }
}

impl rusqlite::ToSql for ReviewStatus {
    fn to_sql(&self) -> rusqlite::Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::from(self.as_str()))
    }
}

impl rusqlite::types::FromSql for ReviewStatus {
    fn column_result(value: rusqlite::types::ValueRef<'_>) -> rusqlite::types::FromSqlResult<Self> {
        match value {
            rusqlite::types::ValueRef::Text(s) => Self::from_str(std::str::from_utf8(s).unwrap_or(""))
                .ok_or_else(|| rusqlite::types::FromSqlError::InvalidType),
            _ => Err(rusqlite::types::FromSqlError::InvalidType),
        }
    }
}

/// Review data model
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReviewData {
    pub id: i64,
    pub original_text: String,
    pub text_hash: String,
    pub predicted_names: Vec<String>,
    pub confidence: f64,
    pub status: ReviewStatus,
    pub corrected_names: Option<Vec<String>>,
    pub review_note: Option<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl ReviewData {
    fn from_row(row: &Row<'_>) -> rusqlite::Result<Self> {
        let predicted_names_str: String = row.get(4)?;
        let corrected_names_str: Option<String> = row.get(7)?;
        let created_at_str: String = row.get(8)?;
        let updated_at_str: String = row.get(9)?;
        
        Ok(Self {
            id: row.get(0)?,
            original_text: row.get(1)?,
            text_hash: row.get(2)?,
            confidence: row.get(3)?,
            predicted_names: serde_json::from_str(&predicted_names_str).unwrap_or_default(),
            status: row.get(5)?,
            review_note: row.get(6)?,
            corrected_names: corrected_names_str
                .and_then(|s| serde_json::from_str(&s).ok()),
            created_at: DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: DateTime::parse_from_rfc3339(&updated_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }
}

/// Compute SHA256 hash of text
pub fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.trim().as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Database repository (thread-safe)
pub struct Repository {
    conn: Mutex<Connection>,
}

impl Repository {
    /// Create a new repository with database connection
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;
        
        // Create table if not exists
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS review_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT NOT NULL,
                text_hash VARCHAR(64) UNIQUE NOT NULL,
                confidence REAL NOT NULL,
                predicted_names TEXT,
                status VARCHAR(20) DEFAULT 'pending',
                review_note TEXT,
                corrected_names TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_status ON review_data(status);
            CREATE INDEX IF NOT EXISTS idx_text_hash ON review_data(text_hash);
            "#,
        )?;
        
        tracing::info!("Database initialized");
        
        Ok(Self { conn: Mutex::new(conn) })
    }
    
    /// Find review by text hash
    pub fn find_by_hash(&self, hash: &str) -> Result<Option<ReviewData>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, original_text, text_hash, confidence, predicted_names, status, review_note, corrected_names, created_at, updated_at FROM review_data WHERE text_hash = ?"
        )?;
        
        let result = stmt.query_row(params![hash], ReviewData::from_row);
        
        match result {
            Ok(data) => Ok(Some(data)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    
    /// Create a new review record
    pub fn create(&self, data: &NewReviewData) -> Result<i64> {
        let predicted_names = serde_json::to_string(&data.predicted_names)?;
        let now = Utc::now();
        let conn = self.conn.lock().unwrap();
        
        conn.execute(
            "INSERT INTO review_data (original_text, text_hash, confidence, predicted_names, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                data.original_text,
                data.text_hash,
                data.confidence,
                predicted_names,
                ReviewStatus::Pending,
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        )?;
        
        Ok(conn.last_insert_rowid())
    }
    
    /// Create if not exists, returns (review, is_new)
    pub fn create_if_not_exists(&self, data: &NewReviewData) -> Result<(ReviewData, bool)> {
        // Check if exists
        if let Some(existing) = self.find_by_hash(&data.text_hash)? {
            return Ok((existing, false));
        }
        
        // Create new
        let id = self.create(data)?;
        let review = self.find_by_id(id)?.unwrap();
        Ok((review, true))
    }
    
    /// Find by ID
    pub fn find_by_id(&self, id: i64) -> Result<Option<ReviewData>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, original_text, text_hash, confidence, predicted_names, status, review_note, corrected_names, created_at, updated_at FROM review_data WHERE id = ?"
        )?;
        
        let result = stmt.query_row(params![id], ReviewData::from_row);
        
        match result {
            Ok(data) => Ok(Some(data)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
    
    /// Get pending reviews
    pub fn get_pending(&self, limit: usize) -> Result<Vec<ReviewData>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, original_text, text_hash, confidence, predicted_names, status, review_note, corrected_names, created_at, updated_at 
             FROM review_data WHERE status = 'pending' 
             ORDER BY created_at DESC LIMIT ?"
        )?;
        
        let reviews = stmt
            .query_map(params![limit as i64], ReviewData::from_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        
        Ok(reviews)
    }
    
    /// Get all reviews with optional filter
    pub fn get_all(&self, status: Option<ReviewStatus>, limit: usize) -> Result<Vec<ReviewData>> {
        let conn = self.conn.lock().unwrap();
        
        let sql = match status {
            Some(_) => "SELECT id, original_text, text_hash, confidence, predicted_names, status, review_note, corrected_names, created_at, updated_at FROM review_data WHERE status = ?1 ORDER BY created_at DESC LIMIT ?2",
            None => "SELECT id, original_text, text_hash, confidence, predicted_names, status, review_note, corrected_names, created_at, updated_at FROM review_data ORDER BY created_at DESC LIMIT ?2",
        };
        
        let mut stmt = conn.prepare(sql)?;
        
        let reviews = match status {
            Some(s) => stmt
                .query_map(params![s, limit as i64], ReviewData::from_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?,
            None => stmt
                .query_map(params![limit as i64], ReviewData::from_row)?
                .collect::<std::result::Result<Vec<_>, _>>()?,
        };
        
        Ok(reviews)
    }
    
    /// Update review status
    pub fn update_status(
        &self,
        id: i64,
        status: ReviewStatus,
        corrected_names: Option<&[String]>,
        note: Option<&str>,
    ) -> Result<()> {
        let corrected_json = corrected_names.map(|n| serde_json::to_string(n).unwrap_or_default());
        let now = Utc::now();
        let conn = self.conn.lock().unwrap();
        
        conn.execute(
            "UPDATE review_data SET status = ?1, corrected_names = ?2, review_note = ?3, updated_at = ?4 WHERE id = ?5",
            params![status, corrected_json, note, now.to_rfc3339(), id],
        )?;
        
        Ok(())
    }
    
    /// Get statistics
    pub fn get_stats(&self) -> Result<serde_json::Value> {
        let conn = self.conn.lock().unwrap();
        let total: i64 = conn.query_row("SELECT COUNT(*) FROM review_data", [], |r| r.get(0))?;
        let pending: i64 = conn.query_row("SELECT COUNT(*) FROM review_data WHERE status = 'pending'", [], |r| r.get(0))?;
        let approved: i64 = conn.query_row("SELECT COUNT(*) FROM review_data WHERE status = 'approved'", [], |r| r.get(0))?;
        let corrected: i64 = conn.query_row("SELECT COUNT(*) FROM review_data WHERE status = 'corrected'", [], |r| r.get(0))?;
        let rejected: i64 = conn.query_row("SELECT COUNT(*) FROM review_data WHERE status = 'rejected'", [], |r| r.get(0))?;
        
        Ok(serde_json::json!({
            "total": total,
            "pending": pending,
            "approved": approved,
            "corrected": corrected,
            "rejected": rejected
        }))
    }
    
    /// Get export data (approved and corrected)
    pub fn get_export_data(&self) -> Result<Vec<ReviewData>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, original_text, text_hash, confidence, predicted_names, status, review_note, corrected_names, created_at, updated_at 
             FROM review_data WHERE status IN ('approved', 'corrected')
             ORDER BY created_at DESC"
        )?;
        
        let reviews = stmt
            .query_map([], ReviewData::from_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        
        Ok(reviews)
    }
}

/// New review data for insertion
pub struct NewReviewData {
    pub original_text: String,
    pub text_hash: String,
    pub predicted_names: Vec<String>,
    pub confidence: f64,
}

/// Thread-safe repository wrapper
#[derive(Clone)]
pub struct RepositorySync(Arc<Repository>);

impl RepositorySync {
    pub fn new(repo: Repository) -> Self {
        Self(Arc::new(repo))
    }
    
    pub fn inner(&self) -> &Repository {
        &self.0
    }
}
