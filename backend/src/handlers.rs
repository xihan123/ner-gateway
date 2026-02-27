//! API handlers

use axum::{
    extract::{Path, Query, State},
    http::header,
    response::{IntoResponse, Response},
    Json,
};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::db::{hash_text, NewReviewData, RepositorySync, ReviewData, ReviewFilter, ReviewStatus};
use crate::engine::NEREngineSync;
use crate::error::{AppError, Result};

/// Application state
#[derive(Clone)]
pub struct AppState {
    pub engine: NEREngineSync,
    pub repo: RepositorySync,
}

/// Extract request body
#[derive(Debug, Deserialize)]
pub struct ExtractRequest {
    pub text: String,
}

/// Extract response
#[derive(Debug, Serialize)]
pub struct ExtractResponse {
    pub names: Vec<String>,
    pub confidence: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub review_id: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_duplicate: Option<bool>,
}

/// Review item for API response
#[derive(Debug, Serialize)]
pub struct ReviewItem {
    pub id: i64,
    pub original_text: String,
    pub predicted_names: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub corrected_names: Option<Vec<String>>,
    pub confidence: f64,
    pub status: String,
    pub created_at: String,
}

impl From<ReviewData> for ReviewItem {
    fn from(data: ReviewData) -> Self {
        Self {
            id: data.id,
            original_text: data.original_text,
            predicted_names: data.predicted_names,
            corrected_names: data.corrected_names,
            confidence: data.confidence,
            status: data.status.as_str().to_string(),
            created_at: data.created_at.to_rfc3339(),
        }
    }
}

/// Review update request
#[derive(Debug, Deserialize)]
pub struct ReviewUpdateRequest {
    pub action: String,
    #[serde(default)]
    pub names: Vec<String>,
    #[serde(default)]
    pub note: Option<String>,
}

/// Query parameters for reviews list
#[derive(Debug, Deserialize)]
pub struct ReviewsQuery {
    pub status: Option<String>,
    /// Minimum confidence (0.0 - 1.0)
    pub confidence_min: Option<f64>,
    /// Maximum confidence (0.0 - 1.0)
    pub confidence_max: Option<f64>,
    /// Filter by whether names were extracted (true = has names, false = no names)
    pub has_names: Option<bool>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
}

fn default_limit() -> usize {
    100
}

/// Query parameters for export
#[derive(Debug, Deserialize)]
pub struct ExportQuery {
    pub format: Option<String>,
}

/// POST /api/extract - Extract names from text
pub async fn extract(
    State(state): State<AppState>,
    Json(req): Json<ExtractRequest>,
) -> Result<Json<ExtractResponse>> {
    let text = req.text.trim();
    
    if text.is_empty() {
        return Err(AppError::BadRequest("text cannot be empty".into()));
    }
    
    let text_hash = hash_text(text);
    let repo = state.repo.inner();
    
    // Check if already exists
    if let Some(existing) = repo.find_by_hash(&text_hash)? {
        return Ok(Json(ExtractResponse {
            names: existing.predicted_names,
            confidence: existing.confidence,
            review_id: Some(existing.id),
            is_duplicate: Some(true),
        }));
    }
    
    // Run inference
    let result = state.engine.predict(text)?;
    
    // Save to database
    let new_review = NewReviewData {
        original_text: text.to_string(),
        text_hash,
        predicted_names: result.names.clone(),
        confidence: result.confidence,
    };
    
    let (review, _) = repo.create_if_not_exists(&new_review)?;
    
    Ok(Json(ExtractResponse {
        names: result.names,
        confidence: result.confidence,
        review_id: Some(review.id),
        is_duplicate: Some(false),
    }))
}

/// GET /api/reviews - Get pending reviews (backward compatible)
pub async fn get_reviews(State(state): State<AppState>) -> Result<Json<serde_json::Value>> {
    let reviews = state.repo.inner().get_pending(50)?;
    let items: Vec<ReviewItem> = reviews.into_iter().map(ReviewItem::from).collect();
    
    Ok(Json(serde_json::json!({
        "count": items.len(),
        "items": items
    })))
}

/// GET /api/reviews/filter - Get reviews with advanced filtering
pub async fn get_filtered_reviews(
    State(state): State<AppState>,
    Query(query): Query<ReviewsQuery>,
) -> Result<Json<serde_json::Value>> {
    // Validate and clamp limit
    let limit = query.limit.min(500).max(1);
    let offset = query.offset;
    
    let filter = ReviewFilter {
        status: query.status.and_then(|s| ReviewStatus::from_str(&s)),
        confidence_min: query.confidence_min,
        confidence_max: query.confidence_max,
        has_names: query.has_names,
        limit,
        offset,
    };
    
    let reviews = state.repo.inner().get_filtered(&filter)?;
    let items: Vec<ReviewItem> = reviews.into_iter().map(ReviewItem::from).collect();
    
    // Get total count for pagination info
    let total_count = state.repo.inner().count_filtered(&filter)?;
    
    Ok(Json(serde_json::json!({
        "count": items.len(),
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + items.len()) < total_count,
        "items": items
    })))
}

/// GET /api/reviews/all - Get all reviews
pub async fn get_all_reviews(
    State(state): State<AppState>,
    Query(query): Query<ReviewsQuery>,
) -> Result<Json<serde_json::Value>> {
    // Validate and clamp limit
    let limit = query.limit.min(500).max(1);
    let offset = query.offset;
    
    let status = query.status.and_then(|s| ReviewStatus::from_str(&s));
    let reviews = state.repo.inner().get_all(status, limit)?;
    let items: Vec<ReviewItem> = reviews.into_iter().map(ReviewItem::from).collect();
    
    // Get total count for pagination info
    let total_count = state.repo.inner().count_all(status)?;
    
    Ok(Json(serde_json::json!({
        "count": items.len(),
        "total": total_count,
        "offset": offset,
        "limit": limit,
        "has_more": (offset + items.len()) < total_count,
        "items": items
    })))
}

/// POST /api/reviews/:id - Update review
pub async fn update_review(
    State(state): State<AppState>,
    Path(id): Path<i64>,
    Json(req): Json<ReviewUpdateRequest>,
) -> Result<Json<serde_json::Value>> {
    let repo = state.repo.inner();
    
    // Find review
    let review = repo
        .find_by_id(id)?
        .ok_or_else(|| AppError::NotFound("review not found".into()))?;
    
    if review.status != ReviewStatus::Pending {
        return Err(AppError::BadRequest("review already processed".into()));
    }
    
    // Parse action
    let (status, corrected_names) = match req.action.as_str() {
        "approve" => (ReviewStatus::Approved, Some(review.predicted_names.as_slice())),
        "correct" => {
            (ReviewStatus::Corrected, Some(req.names.as_slice()))
        }
        "reject" => (ReviewStatus::Rejected, None),
        _ => return Err(AppError::BadRequest("invalid action: must be approve, correct, or reject".into())),
    };
    
    repo.update_status(id, status, corrected_names, req.note.as_deref())?;
    
    Ok(Json(serde_json::json!({
        "id": id,
        "status": status.as_str(),
        "message": "review updated successfully"
    })))
}

/// GET /api/stats - Get statistics
pub async fn get_stats(State(state): State<AppState>) -> Result<Json<serde_json::Value>> {
    let stats = state.repo.inner().get_stats()?;
    Ok(Json(stats))
}

/// GET /api/export - Export training data
pub async fn export_data(
    State(state): State<AppState>,
    Query(query): Query<ExportQuery>,
) -> Result<Response> {
    let reviews = state.repo.inner().get_export_data()?;
    
    // Convert to BIO format
    let export_items: Vec<serde_json::Value> = reviews
        .iter()
        .map(|r| {
            let names = if r.status == ReviewStatus::Corrected {
                r.corrected_names.clone().unwrap_or_default()
            } else {
                r.predicted_names.clone()
            };
            
            let (tokens, labels) = text_to_bio_tags(&r.original_text, &names);
            
            serde_json::json!({
                "tokens": tokens,
                "labels": labels
            })
        })
        .collect();
    
    if query.format.as_deref() == Some("jsonl") || query.format.as_deref() == Some("bio") {
        // Return JSONL format
        let body = export_items
            .iter()
            .map(|item| serde_json::to_string(item).unwrap_or_default())
            .collect::<Vec<_>>()
            .join("\n");
        
        return Ok((
            [
                (header::CONTENT_TYPE, "application/x-ndjson"),
                (
                    header::CONTENT_DISPOSITION,
                    "attachment; filename=training_data.jsonl",
                ),
            ],
            body,
        )
            .into_response());
    }
    
    // Default JSON response
    Ok(Json(serde_json::json!({
        "count": export_items.len(),
        "data": export_items
    }))
    .into_response())
}

/// GET /health - Health check
pub async fn health_check(State(state): State<AppState>) -> Result<Json<serde_json::Value>> {
    let repo = state.repo.inner();
    let db_status = match repo.get_stats() {
        Ok(_) => "connected",
        Err(e) => return Ok(Json(serde_json::json!({
            "status": "degraded",
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "model": "loaded",
            "db": format!("error: {}", e)
        }))),
    };
    
    Ok(Json(serde_json::json!({
        "status": "ok",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "model": "loaded",
        "db": db_status
    })))
}

/// Check if a name is valid (contains non-whitespace, non-punctuation characters)
fn is_valid_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }
    
    name.chars().any(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
}

/// Clean name by removing whitespace and invalid characters
fn clean_name_for_bio(name: &str) -> String {
    name.chars()
        .filter(|c| !c.is_whitespace() && !c.is_ascii_punctuation())
        .collect()
}

/// Convert text and names to BIO tags
fn text_to_bio_tags(text: &str, names: &[String]) -> (Vec<String>, Vec<String>) {
    // Clean HTML tags
    let clean_text = clean_html(text);
    
    // Tokenize: each character as a token
    let tokens: Vec<char> = clean_text.chars().collect();
    let mut labels = vec!["O"; tokens.len()];
    
    // Set BIO tags for each name
    for name in names {
        // Clean and validate name
        let cleaned_name = clean_name_for_bio(name);
        if !is_valid_name(&cleaned_name) {
            continue;
        }
        
        let name_chars: Vec<char> = cleaned_name.chars().collect();
        if name_chars.is_empty() {
            continue;
        }
        
        // Find name in text
        for i in 0..=(tokens.len().saturating_sub(name_chars.len())) {
            let match_found = name_chars
                .iter()
                .enumerate()
                .all(|(j, &c)| tokens.get(i + j) == Some(&c));
            
            if match_found {
                labels[i] = "B-PER";
                for j in 1..name_chars.len() {
                    labels[i + j] = "I-PER";
                }
            }
        }
    }
    
    let token_strs: Vec<String> = tokens.iter().map(|c| c.to_string()).collect();
    let label_strs: Vec<String> = labels.iter().map(|s| s.to_string()).collect();
    
    (token_strs, label_strs)
}

/// Remove HTML tags
fn clean_html(text: &str) -> String {
    let re = Regex::new(r"<[^>]+>").unwrap();
    re.replace_all(text, "").trim().to_string()
}

