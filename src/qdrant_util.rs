use anyhow::Result;
use qdrant_client::qdrant::{
    CreateCollectionBuilder, Distance, PointStruct, ScalarQuantizationBuilder, SearchParamsBuilder,
    SearchPointsBuilder, SearchResponse, UpsertPointsBuilder, VectorParamsBuilder,
};
use qdrant_client::Qdrant;

pub async fn collection_exists(qdrant_client: &Qdrant, collection_name: &str) -> Result<bool> {
    Ok(qdrant_client.collection_exists(collection_name).await?)
}

pub async fn create_collection_if_not_exists(
    qdrant_client: &Qdrant,
    collection_name: &str,
    vector_size: u64,
    distance: Distance,
) -> Result<()> {
    if collection_exists(qdrant_client, collection_name).await? {
        println!(
            "'{}' collection already exists in Qdrant; do not create",
            collection_name
        );
    } else {
        qdrant_client
            .create_collection(
                CreateCollectionBuilder::new(collection_name)
                    .vectors_config(VectorParamsBuilder::new(vector_size, distance))
                    .quantization_config(ScalarQuantizationBuilder::default()),
            )
            .await?;
        println!("created collection '{}' in Qdrant", collection_name);
    }
    Ok(())
}

pub async fn upsert_points(
    qdrant_client: &Qdrant,
    collection_name: &str,
    points: Vec<PointStruct>,
) -> Result<bool> {
    qdrant_client
        .upsert_points(UpsertPointsBuilder::new(collection_name, points))
        .await?;
    Ok(true)
}

pub async fn search_points(
    qdrant_client: &Qdrant,
    collection_name: &str,
    embedding_vec: Vec<f32>,
    vector_size: u64,
    limit: u64,
) -> Result<SearchResponse> {
    let search_result = qdrant_client
        .search_points(
            SearchPointsBuilder::new(collection_name, embedding_vec, vector_size)
                .with_payload(true)
                .params(SearchParamsBuilder::default().exact(true))
                .limit(limit),
        )
        .await?;
    Ok(search_result)
}
