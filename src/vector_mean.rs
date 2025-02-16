use ndarray::{Array, Array2, Axis};
use ndarray_stats::SummaryStatisticsExt;

// Example embeddings: a vector of vectors
// let embeddings: Vec<Vec<f32>> = vec![
//     vec![0.1, 0.2, 0.3],
//     vec![0.4, 0.5, 0.6],
//     vec![0.7, 0.8, 0.9],
//     // Add more embeddings as needed
// ];
pub fn mean(embeddings: Vec<Vec<f32>>, weights: Vec<f32>) -> Vec<f32> {
    // Convert Vec<Vec<f32>> to a 2D ndarray
    let num_embeddings = embeddings.len();
    let embedding_dim = embeddings[0].len();
    let flat_embeddings: Vec<f32> = embeddings.into_iter().flatten().collect();
    let array = Array2::from_shape_vec((num_embeddings, embedding_dim), flat_embeddings)
        .expect("Error creating ndarray");

    let weights = Array::from_vec(weights);
    let mean_embedding = array
        .weighted_mean_axis(Axis(0), &weights)
        .expect("Error computing mean");

    // `mean_embedding` is a 1D array representing the average embedding
    let (v, _) = mean_embedding.into_raw_vec_and_offset();
    v
}
