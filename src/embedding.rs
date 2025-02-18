use crate::splitter;
use crate::vector_mean;
use anyhow::Result;
use fastembed::TextEmbedding;
use tokenizers::tokenizer::Tokenizer;

pub fn embed(text: &str, embedding_model: &TextEmbedding, tokenizer: &Tokenizer) -> Vec<f32> {
    let chunks: Vec<String> = splitter::split(text, tokenizer);
    let embedding: Vec<Vec<f32>> = chunks
        .iter()
        .map(|s| embed_a_chunk(s, embedding_model).unwrap())
        .collect();
    let weights: Vec<f32> = chunks.iter().map(|s| s.len() as f32).collect();
    vector_mean::mean(embedding, weights)
}

fn embed_a_chunk(text: &str, embedding_model: &TextEmbedding) -> Result<Vec<f32>> {
    let embedding = embedding_model.embed(vec![text], None).unwrap();
    Ok(embedding[0].to_vec())
}
