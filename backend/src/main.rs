//! NER Gateway - Rust implementation
//! 
//! A Named Entity Recognition inference gateway with ONNX Runtime,
//! SQLite persistence, and REST API endpoints.

mod config;
mod db;
mod engine;
mod error;
mod handlers;
mod tokenizer;

use std::net::SocketAddr;
use std::path::Path;

use axum::{
    routing::{get, post},
    Router,
};
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::{ServeDir, ServeFile};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::config::Config;
use crate::db::{Repository, RepositorySync};
use crate::engine::{NEREngine, NEREngineSync};
use crate::handlers::AppState;

/// Find index.html in multiple possible locations
/// Priority:
/// 1. ./index.html (Docker container: /app/index.html, or running from backend dir)
/// 2. backend/index.html (Running from project root)
/// 3. ../index.html (Running from backend/target/debug dir during development)
fn find_static_files_dir() -> &'static str {
    // Try multiple paths in order
    let candidates = [
        "./index.html",           // Docker or running from backend/
        "backend/index.html",     // Running from project root
        "../index.html",          // Running from backend/target/debug/
    ];
    
    for path in candidates {
        if Path::new(path).exists() {
            let dir = path.rfind('/').map(|i| &path[..i]).unwrap_or(".");
            tracing::info!("Found static files at: {}", dir);
            return Box::leak(dir.to_string().into_boxed_str());
        }
    }
    
    // Default to current directory (Docker default)
    tracing::warn!("index.html not found, using default path: ./");
    "."
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ner_gateway=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    // Load configuration
    let config = Config::from_env();
    tracing::info!("Configuration: {:?}", config);
    
    // Initialize NER engine
    tracing::info!("Initializing NER engine...");
    let engine = NEREngine::new(&config.model_path, &config.vocab_path)?;
    let engine_sync = NEREngineSync::new(engine);
    
    // Initialize database
    tracing::info!("Initializing database...");
    let repo = Repository::new(&config.db_path)?;
    let repo_sync = RepositorySync::new(repo);
    
    // Create app state
    let state = AppState {
        engine: engine_sync,
        repo: repo_sync,
    };
    
    // Find static files directory (compatible with Docker, project root, and backend dir)
    let static_dir = find_static_files_dir();
    
    // Build router
    let app = Router::new()
        // API routes
        .route("/api/extract", post(handlers::extract))
        .route("/api/reviews", get(handlers::get_reviews))
        .route("/api/reviews/filter", get(handlers::get_filtered_reviews))
        .route("/api/reviews/all", get(handlers::get_all_reviews))
        .route("/api/reviews/{id}", post(handlers::update_review))
        .route("/api/stats", get(handlers::get_stats))
        .route("/api/export", get(handlers::export_data))
        .route("/health", get(handlers::health_check))
        // Static files - serve index.html for root and fallback for SPA
        .fallback_service(
            ServeDir::new(static_dir)
                .fallback(ServeFile::new(format!("{}/index.html", static_dir)))
        )
        // Layers
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    
    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    tracing::info!("Server starting on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;
    
    tracing::info!("Server stopped");
    Ok(())
}

/// Graceful shutdown signal
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };
    
    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };
    
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    
    tracing::info!("Shutdown signal received");
}
