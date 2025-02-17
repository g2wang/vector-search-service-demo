use anyhow::Result;
use fastembed::{
    read_file_to_bytes, InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use std::path::PathBuf;

pub fn get_model() -> Result<TextEmbedding> {
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
