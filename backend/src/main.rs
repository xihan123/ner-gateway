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

fn find_static_files_dir() -> &'static str {
    let candidates = ["./index.html", "backend/index.html", "../index.html"];
    
    for path in candidates {
        if Path::new(path).exists() {
            let dir = path.rfind('/').map(|i| &path[..i]).unwrap_or(".");
            tracing::info!("Found static files at: {}", dir);
            return Box::leak(dir.to_string().into_boxed_str());
        }
    }
    
    tracing::warn!("index.html not found, using default path: ./");
    "."
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "ner_gateway=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();
    
    let config = Config::from_env();
    tracing::info!("Config: {:?}", config);
    
    let engine = NEREngine::new(&config.model_path, &config.vocab_path)?;
    let engine_sync = NEREngineSync::new(engine);
    
    let repo = Repository::new(&config.db_path)?;
    let repo_sync = RepositorySync::new(repo);
    
    let state = AppState {
        engine: engine_sync,
        repo: repo_sync,
    };
    
    let static_dir = find_static_files_dir();
    
    let app = Router::new()
        .route("/api/extract", post(handlers::extract))
        .route("/api/extract/batch", post(handlers::batch_extract))
        .route("/api/reviews", get(handlers::get_reviews))
        .route("/api/reviews/filter", get(handlers::get_filtered_reviews))
        .route("/api/reviews/all", get(handlers::get_all_reviews))
        .route("/api/reviews/{id}", post(handlers::update_review))
        .route("/api/stats", get(handlers::get_stats))
        .route("/api/export", get(handlers::export_data))
        .route("/api/gpu", get(handlers::gpu_status))
        .route("/health", get(handlers::health_check))
        .fallback_service(
            ServeDir::new(static_dir)
                .fallback(ServeFile::new(format!("{}/index.html", static_dir)))
        )
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);
    
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    tracing::info!("Listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal())
    .await?;
    
    Ok(())
}

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
}
