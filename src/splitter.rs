use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::tokenizer::Tokenizer;

const MAX_TOKENS_PER_CHUNK: usize = 256; // all-MiniLM-L6-v2 embedding max tokens limit

pub fn split(text: &str, tokenizer: &Tokenizer) -> Vec<String> {
    let splitter = TextSplitter::new(ChunkConfig::new(MAX_TOKENS_PER_CHUNK).with_sizer(tokenizer));
    splitter.chunks(text).map(|s| s.to_string()).collect()
}
