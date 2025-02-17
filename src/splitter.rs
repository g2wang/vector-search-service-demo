use crate::tokenizer_factory;
use text_splitter::{ChunkConfig, TextSplitter};

const MAX_TOKENS_PER_CHUNK: usize = 256; // all-MiniLM-L6-v2 embedding max tokens limit

pub fn split(text: &str) -> Vec<String> {
    let tokenizer = tokenizer_factory::get_tokenizer();
    let splitter = TextSplitter::new(ChunkConfig::new(MAX_TOKENS_PER_CHUNK).with_sizer(tokenizer));
    splitter.chunks(text).map(|s| s.to_string()).collect()
}
