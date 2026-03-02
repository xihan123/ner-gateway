use std::collections::HashSet;
use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use rusqlite::{params, types::ToSqlOutput, Connection, Row};
use sha2::{Digest, Sha256};

use crate::error::Result;

const HASH_CACHE_MAX_SIZE: usize = 100_000;

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
            corrected_names: corrected_names_str.and_then(|s| serde_json::from_str(&s).ok()),
            created_at: DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
            updated_at: DateTime::parse_from_rfc3339(&updated_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .unwrap_or_else(|_| Utc::now()),
        })
    }
}

pub fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.trim().as_bytes());
    format!("{:x}", hasher.finalize())
}

pub struct Repository {
    conn: Mutex<Connection>,
    hash_cache: Mutex<HashSet<String>>,
}

impl Repository {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path)?;

        conn.execute_batch(
            r#"
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-64000;
            PRAGMA temp_store=MEMORY;
            PRAGMA journal_size_limit=104857600;
            "#,
        )?;
        
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

        let mut hash_cache = HashSet::new();
        {
            let mut stmt = conn.prepare("SELECT text_hash FROM review_data")?;
            let hashes: Vec<String> = stmt
                .query_map([], |row| row.get(0))?
                .collect::<std::result::Result<Vec<_>, _>>()?;

            for hash in hashes {
                hash_cache.insert(hash);
            }
        }

        tracing::info!("Database initialized with WAL mode, {} hashes cached", hash_cache.len());

        Ok(Self {
            conn: Mutex::new(conn),
            hash_cache: Mutex::new(hash_cache),
        })
    }

    fn is_hash_cached(&self, hash: &str) -> bool {
        let cache = self.hash_cache.lock().unwrap();
        cache.contains(hash)
    }

    fn cache_hash(&self, hash: String) {
        let mut cache = self.hash_cache.lock().unwrap();

        if cache.len() >= HASH_CACHE_MAX_SIZE {
            let to_remove: Vec<String> = cache.iter().take(cache.len() / 2).cloned().collect();
            for h in to_remove {
                cache.remove(&h);
            }
            tracing::warn!("Hash cache reached max size, evicted {} entries", cache.len() / 2);
        }

        cache.insert(hash);
    }
    
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

    #[allow(dead_code)]
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
    
    pub fn create_if_not_exists(&self, data: &NewReviewData) -> Result<(ReviewData, bool)> {
        if self.is_hash_cached(&data.text_hash) {
            let existing = self.find_by_hash(&data.text_hash)?.unwrap();
            return Ok((existing, false));
        }

        let predicted_names = serde_json::to_string(&data.predicted_names)?;
        let now = Utc::now();
        let conn = self.conn.lock().unwrap();

        let result = conn.execute(
            "INSERT INTO review_data (original_text, text_hash, confidence, predicted_names, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(text_hash) DO NOTHING",
            params![
                data.original_text,
                data.text_hash,
                data.confidence,
                predicted_names,
                ReviewStatus::Pending,
                now.to_rfc3339(),
                now.to_rfc3339(),
            ],
        );

        match result {
            Ok(rows_affected) => {
                if rows_affected > 0 {
                    let id = conn.last_insert_rowid();
                    drop(conn);

                    self.cache_hash(data.text_hash.clone());
                    let review = self.find_by_id(id)?.unwrap();
                    Ok((review, true))
                } else {
                    let hash = data.text_hash.clone();
                    drop(conn);

                    self.cache_hash(hash);
                    let existing = self.find_by_hash(&data.text_hash)?.unwrap();
                    Ok((existing, false))
                }
            }
            Err(e) => Err(e.into()),
        }
    }
    
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
    
    pub fn count_all(&self, status: Option<ReviewStatus>) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        
        let sql = match status {
            Some(_) => "SELECT COUNT(*) FROM review_data WHERE status = ?1",
            None => "SELECT COUNT(*) FROM review_data",
        };
        
        let count: i64 = match status {
            Some(s) => conn.query_row(sql, params![s], |r| r.get(0))?,
            None => conn.query_row(sql, [], |r| r.get(0))?,
        };
        
        Ok(count as usize)
    }
    
    pub fn get_filtered(&self, filter: &ReviewFilter) -> Result<Vec<ReviewData>> {
        let conn = self.conn.lock().unwrap();
        
        let mut sql = String::from(
            "SELECT id, original_text, text_hash, confidence, predicted_names, status, review_note, corrected_names, created_at, updated_at FROM review_data WHERE 1=1"
        );
        let mut param_values: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        
        if let Some(status) = &filter.status {
            sql.push_str(" AND status = ?");
            param_values.push(Box::new(status.as_str().to_string()));
        }
        
        if let Some(min) = filter.confidence_min {
            sql.push_str(" AND confidence >= ?");
            param_values.push(Box::new(min));
        }
        if let Some(max) = filter.confidence_max {
            sql.push_str(" AND confidence <= ?");
            param_values.push(Box::new(max));
        }
        
        if let Some(has_names) = filter.has_names {
            if has_names {
                sql.push_str(" AND predicted_names != '[]' AND predicted_names IS NOT NULL");
            } else {
                sql.push_str(" AND (predicted_names = '[]' OR predicted_names IS NULL)");
            }
        }
        
        sql.push_str(" ORDER BY created_at DESC LIMIT ? OFFSET ?");
        param_values.push(Box::new(filter.limit as i64));
        param_values.push(Box::new(filter.offset as i64));
        
        let params: Vec<&dyn rusqlite::ToSql> = param_values.iter().map(|v| v.as_ref()).collect();
        
        let mut stmt = conn.prepare(&sql)?;
        let reviews = stmt
            .query_map(params.as_slice(), ReviewData::from_row)?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        
        Ok(reviews)
    }
    
    pub fn count_filtered(&self, filter: &ReviewFilter) -> Result<usize> {
        let conn = self.conn.lock().unwrap();
        
        let mut sql = String::from("SELECT COUNT(*) FROM review_data WHERE 1=1");
        let mut param_values: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        
        if let Some(status) = &filter.status {
            sql.push_str(" AND status = ?");
            param_values.push(Box::new(status.as_str().to_string()));
        }
        
        if let Some(min) = filter.confidence_min {
            sql.push_str(" AND confidence >= ?");
            param_values.push(Box::new(min));
        }
        if let Some(max) = filter.confidence_max {
            sql.push_str(" AND confidence <= ?");
            param_values.push(Box::new(max));
        }
        
        if let Some(has_names) = filter.has_names {
            if has_names {
                sql.push_str(" AND predicted_names != '[]' AND predicted_names IS NOT NULL");
            } else {
                sql.push_str(" AND (predicted_names = '[]' OR predicted_names IS NULL)");
            }
        }
        
        let params: Vec<&dyn rusqlite::ToSql> = param_values.iter().map(|v| v.as_ref()).collect();
        
        let count: i64 = conn.query_row(&sql, params.as_slice(), |r| r.get(0))?;
        
        Ok(count as usize)
    }
    
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

pub struct NewReviewData {
    pub original_text: String,
    pub text_hash: String,
    pub predicted_names: Vec<String>,
    pub confidence: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ReviewFilter {
    pub status: Option<ReviewStatus>,
    pub confidence_min: Option<f64>,
    pub confidence_max: Option<f64>,
    pub has_names: Option<bool>,
    pub limit: usize,
    pub offset: usize,
}

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
