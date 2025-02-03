use anyhow::Result;
use axum::{extract::State, routing::post, Json, Router};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, ScalarQuantizationBuilder, SearchParamsBuilder,
    SearchPointsBuilder, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::{qdrant::PointStruct, Payload, Qdrant};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

const COLLECTION_NAME: &str = "tips";
const VECTOR_SIZE: u64 = 384; // all-MiniLM-L6-v2 embedding Size: 384 dimensions

// Request/Response structs
#[derive(Deserialize, Serialize)]
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
    qdrant_client: Qdrant,
    embedding_model: TextEmbedding,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Initialize the embedding model
    let embedding_model = TextEmbedding::try_new(InitOptions::new(EmbeddingModel::AllMiniLML6V2))?;

    // Initialize Qdrant client
    let qdrant_client = Qdrant::from_url("http://localhost:6334").build()?;

    qdrant_client
        .create_collection(
            CreateCollectionBuilder::new(COLLECTION_NAME)
                .vectors_config(VectorParamsBuilder::new(VECTOR_SIZE, Distance::Cosine))
                .quantization_config(ScalarQuantizationBuilder::default()),
        )
        .await?;

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
    Json(add_tip_request): Json<AddTipRequest>,
) -> Json<serde_json::Value> {
    // Generate embedding
    let embedding = state
        .embedding_model
        .embed(vec![&add_tip_request.text], None)
        .unwrap();
    let embedding_vec = embedding[0].to_vec();

    let payload: Payload = json!(add_tip_request).try_into().unwrap();
    let points = vec![PointStruct::new(
        uuid::Uuid::new_v4().to_string(),
        embedding_vec,
        payload,
    )];

    match state
        .qdrant_client
        .upsert_points(UpsertPointsBuilder::new(COLLECTION_NAME, points))
        .await
    {
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
) -> Json<Vec<SearchResponse>> {
    // Generate embedding for search query
    let embedding = state
        .embedding_model
        .embed(vec![&add_tip_request.text], None)
        .unwrap();
    let embedding_vec = embedding[0].to_vec();

    // Search in Qdrant
    let search_result = state
        .qdrant_client
        .search_points(
            SearchPointsBuilder::new(COLLECTION_NAME, embedding_vec, VECTOR_SIZE)
                .with_payload(true)
                .params(SearchParamsBuilder::default().exact(true)),
        )
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
                        .unwrap_or(&"".to_owned())
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
