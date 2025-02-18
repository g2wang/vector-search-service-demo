mod embedding;
mod qdrant_util;
mod splitter;
mod thread_safty;
mod vector_mean;

use anyhow::Result;
use axum::{extract::State, routing::post, Json, Router};
use dotenv::dotenv;
use fastembed::{
    read_file_to_bytes, InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use qdrant_client::qdrant::{Distance, PointStruct};
use qdrant_client::{Payload, Qdrant};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::tokenizer::Tokenizer;

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
    embedding_model: TextEmbedding,
    tokenizer: Tokenizer,
}

#[tokio::main]
async fn main() -> Result<()> {
    thread_safty::assert_send_sync::<Qdrant>();
    thread_safty::assert_send_sync::<TextEmbedding>();
    thread_safty::assert_send_sync::<Tokenizer>();

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

    let embedding_model = get_model()?;
    let tokenizer = get_tokenizer();

    // Create app state
    let state = Arc::new(AppState {
        qdrant_client,
        embedding_model,
        tokenizer,
    });

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
    let embedding_vec = embedding::embed(
        &add_tip_request.text,
        &state.embedding_model,
        &state.tokenizer,
    );
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
    let embedding_vec = embedding::embed(
        &add_tip_request.text,
        &state.embedding_model,
        &state.tokenizer,
    );
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

fn get_model() -> Result<TextEmbedding> {
    let base_path = PathBuf::from("all-MiniLM-L6-v2");
    let onnx_path = base_path.join("onnx").join("model.onnx");
    let tokenizer_path = base_path.join("tokenizer.json");
    let config_path = base_path.join("config.json");
    let special_tokens_map_path = base_path.join("special_tokens_map.json");
    let tokenizer_config_path = base_path.join("tokenizer_config.json");

    let onnx_bytes = read_file_to_bytes(&onnx_path)?;
    let tokenizer_file = read_file_to_bytes(&tokenizer_path)?;
    let config_file = read_file_to_bytes(&config_path)?;
    let special_tokens_map_file = read_file_to_bytes(&special_tokens_map_path)?;
    let tokenizer_config_file = read_file_to_bytes(&tokenizer_config_path)?;

    let tokenizer_files: TokenizerFiles = TokenizerFiles {
        tokenizer_file,
        config_file,
        special_tokens_map_file,
        tokenizer_config_file,
    };

    let user_model =
        UserDefinedEmbeddingModel::new(onnx_bytes, tokenizer_files).with_pooling(Pooling::Mean); // Optional: Set pooling strategy

    Ok(TextEmbedding::try_new_from_user_defined(
        user_model,
        InitOptionsUserDefined::default(),
    )?)
}

fn get_tokenizer() -> Tokenizer {
    let base_path = PathBuf::from("all-MiniLM-L6-v2");
    let tokenizer_path = base_path.join("tokenizer.json");
    Tokenizer::from_file(tokenizer_path).unwrap()
}
