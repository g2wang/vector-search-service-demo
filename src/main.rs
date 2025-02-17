mod embedding;
mod embedding_model_factory;
mod qdrant_util;
mod splitter;
mod tokenizer_factory;
mod vector_mean;

use anyhow::Result;
use axum::{extract::State, routing::post, Json, Router};
use dotenv::dotenv;
use qdrant_client::qdrant::{Distance, PointStruct};
use qdrant_client::{Payload, Qdrant};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::sync::Arc;

const COLLECTION_NAME: &str = "tips";
const VECTOR_SIZE: u64 = 384; // all-MiniLM-L6-v2 embedding Size: 384 dimensions

// Request/Response structs
#[derive(Deserialize, Serialize)]
struct AddTipRequest {
    text: String,
}

#[derive(Serialize)]
struct SearchResult {
    text: String,
    score: f32,
}

// App state
struct AppState {
    qdrant_client: Qdrant,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Initialize Qdrant client
    let qdrant_client = Qdrant::from_url(&env::var("QDRANT_URL").unwrap()).build()?;

    qdrant_util::create_collection_if_not_exists(
        &qdrant_client,
        COLLECTION_NAME,
        VECTOR_SIZE,
        Distance::Cosine,
    )
    .await?;

    // Create app state
    let state = Arc::new(AppState { qdrant_client });

    // Build router
    let app = Router::new()
        .route("/tip", post(add_tip))
        .route("/search", post(search))
        .with_state(state);

    let server_url = env::var("SERVER_URL").unwrap();
    // Start server
    let listener = tokio::net::TcpListener::bind(&server_url).await?;
    println!("Server running on {}", server_url);
    axum::serve(listener, app).await?;

    Ok(())
}

// Handler for adding tips
async fn add_tip(
    State(state): State<Arc<AppState>>,
    Json(add_tip_request): Json<AddTipRequest>,
) -> Json<serde_json::Value> {
    let embedding_vec = embedding::embed(&add_tip_request.text);
    let payload: Payload = json!(add_tip_request).try_into().unwrap();
    let points = vec![PointStruct::new(
        uuid::Uuid::new_v4().to_string(),
        embedding_vec,
        payload,
    )];

    match qdrant_util::upsert_points(&state.qdrant_client, COLLECTION_NAME, points).await {
        Ok(_) => Json(json!({ "status": "success" })),
        Err(e) => Json(json!({
            "status": "error",
            "message": e.to_string()
        })),
    }
}

// Handler for searching tips
async fn search(
    State(state): State<Arc<AppState>>,
    Json(add_tip_request): Json<AddTipRequest>,
) -> Json<Vec<SearchResult>> {
    let embedding_vec = embedding::embed(&add_tip_request.text);
    // Search in Qdrant
    let search_result = qdrant_util::search_points(
        &state.qdrant_client,
        COLLECTION_NAME,
        embedding_vec,
        VECTOR_SIZE,
        1,
    )
    .await;

    match search_result {
        Ok(results) => {
            let responses: Vec<SearchResult> = results
                .result
                .into_iter()
                .map(|scored_point| {
                    let text = scored_point
                        .payload
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or(&"".to_owned())
                        .to_string();
                    SearchResult {
                        text,
                        score: scored_point.score,
                    }
                })
                .collect();
            Json(responses)
        }
        Err(_) => Json(vec![SearchResult {
            text: "No clue".to_owned(),
            score: 0 as f32,
        }]),
    }
}
