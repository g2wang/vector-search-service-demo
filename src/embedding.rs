use crate::embedding_model_factory;
use crate::splitter;
use crate::vector_mean;
use anyhow::Result;

pub fn embed(text: &str) -> Vec<f32> {
    let chunks: Vec<String> = splitter::split(text);
    let embedding: Vec<Vec<f32>> = chunks.iter().map(|s| embed_a_chunk(s).unwrap()).collect();
    let weights: Vec<f32> = chunks.iter().map(|s| s.len() as f32).collect();
    vector_mean::mean(embedding, weights)
}

fn embed_a_chunk(text: &str) -> Result<Vec<f32>> {
    let embedding_model = embedding_model_factory::get_model()?;
    let embedding = embedding_model.embed(vec![text], None).unwrap();
    Ok(embedding[0].to_vec())
}
