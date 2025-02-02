Please build a RESTful service using Axum, a Rust framework. The service should
do similarity search against a Qdrant vector database. The collection name in
the database should be named "tips". The service should have two endpoints -
one for adding a tip which is a piece of text to the vector database and the
other for querying against the database to get the top 1 match based on
similarity search. For the embedding, please use MPNet.

Answeer from claude 3.5:

I'll help you create a RESTful service using Axum that integrates with Qdrant for vector similarity search.

```rust
[package]
name = "vector-search-service"
version = "0.1.0"
edition = "2021"

[dependencies]
axum = "0.7"
tokio = { version = "1.0", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
qdrant-client = "1.7"
rust-bert = "0.21"
anyhow = "1.0"
tower-http = { version = "0.5", features = ["cors"] }
tracing = "0.1"
tracing-subscriber = "0.3"

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

```

I've created a complete RESTful service using Axum that integrates with Qdrant for vector similarity search. Here are the key components:

1. Dependencies:
   - `axum` for the web framework
   - `qdrant-client` for interacting with Qdrant
   - `rust-bert` for MPNet embeddings
   - Other utility crates for serialization and error handling

2. Endpoints:
   - POST `/tip` - Add a new tip to the database
   - POST `/search` - Search for similar tips

3. Features:
   - Uses MPNet for generating embeddings (768-dimensional vectors)
   - Creates a Qdrant collection named "tips" if it doesn't exist
   - Stores both the original text and embeddings
   - Returns the top match with similarity score

To use this service:

1. Make sure you have a Qdrant instance running (default: http://localhost:6334)

2. Add a tip:
```bash
curl -X POST http://localhost:3000/tip \
  -H "Content-Type: application/json" \
  -d '{"text": "Remember to drink water regularly"}'
```

3. Search for similar tips:
```bash
curl -X POST http://localhost:3000/search \
  -H "Content-Type: application/json" \
  -d '{"text": "Stay hydrated"}'
```

Would you like me to:
1. Add error handling for the embedding generation?
2. Implement batch processing for multiple tips?
3. Add configuration options for the Qdrant connection?
4. Add any other features?
