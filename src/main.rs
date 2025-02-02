use std::sync::Arc;
use axum::{
    routing::{post, get},
    Router,
    Json,
    extract::State,
};
use qdrant_client::prelude::*;
use qdrant_client::qdrant::{
    vectors_config::Config,
    VectorParams,
    VectorsConfig,
    Distance,
};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsModel,
    SentenceEmbeddingsBuilder,
};
use serde::{Serialize, Deserialize};
use anyhow::Result;

// Request/Response structs
#[derive(Deserialize)]
struct AddTipRequest {
    text: String,
}

#[derive(Serialize)]
struct SearchResponse {
    text: String,
    score: f32,
}

// App state
struct AppState {
    qdrant_client: QdrantClient,
    embedding_model: SentenceEmbeddingsModel,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Initialize Qdrant client
    let qdrant_client = QdrantClient::from_url("http://localhost:6334").await?;

    // Create collection if it doesn't exist
    let collection_name = "tips";
    let vector_size = 768; // MPNet embedding size

    let vectors_config = VectorsConfig {
        config: Some(Config::Params(VectorParams {
            size: vector_size,
            distance: Distance::Cosine.into(),
            ..Default::default()
        })),
    };

    match qdrant_client
        .create_collection(&CreateCollection {
            collection_name: collection_name.to_string(),
            vectors_config: Some(vectors_config),
            ..Default::default()
        })
        .await
    {
        Ok(_) => println!("Collection created successfully"),
        Err(e) => println!("Collection creation error (might already exist): {}", e),
    }

    // Initialize MPNet model
    let embedding_model = SentenceEmbeddingsBuilder::remote(
        rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType::MPNet,
    )
    .create_model()?;

    // Create app state
    let state = Arc::new(AppState {
        qdrant_client,
        embedding_model,
    });

    // Build router
    let app = Router::new()
        .route("/tip", post(add_tip))
        .route("/search", post(search))
        .with_state(state);

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    println!("Server running on http://0.0.0.0:3000");
    axum::serve(listener, app).await?;

    Ok(())
}

// Handler for adding tips
async fn add_tip(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<AddTipRequest>,
) -> Json<serde_json::Value> {
    // Generate embedding
    let embedding = state.embedding_model.encode(&[&payload.text]).unwrap();
    let embedding_vec = embedding[0].to_vec();

    // Add point to Qdrant
    let point = PointStruct::new(
        uuid::Uuid::new_v4().to_string(),
        embedding_vec,
        payload.text.clone(),
    );

    match state
        .qdrant_client
        .upsert_points("tips", vec![point], None)
        .await
    {
        Ok(_) => Json(serde_json::json!({ "status": "success" })),
        Err(e) => Json(serde_json::json!({
            "status": "error",
            "message": e.to_string()
        })),
    }
}

// Handler for searching tips
async fn search(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<AddTipRequest>,
) -> Json<Vec<SearchResponse>> {
    // Generate embedding for search query
    let embedding = state.embedding_model.encode(&[&payload.text]).unwrap();
    let embedding_vec = embedding[0].to_vec();

    // Search in Qdrant
    let search_result = state
        .qdrant_client
        .search_points(&SearchPoints {
            collection_name: "tips".to_string(),
            vector: embedding_vec,
            limit: 1,
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await;

    match search_result {
        Ok(results) => {
            let responses: Vec<SearchResponse> = results
                .result
                .into_iter()
                .map(|scored_point| {
                    let text = scored_point
                        .payload
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    SearchResponse {
                        text,
                        score: scored_point.score,
                    }
                })
                .collect();
            Json(responses)
        }
        Err(_) => Json(vec![]),
    }
}

