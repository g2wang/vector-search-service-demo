use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::tokenizer::Tokenizer;

pub fn split(text: &str, tokenizer: &Tokenizer, max_tokens_per_chunk: usize) -> Vec<String> {
    let splitter = TextSplitter::new(ChunkConfig::new(max_tokens_per_chunk).with_sizer(tokenizer));
    splitter.chunks(text).map(|s| s.to_string()).collect()
}
