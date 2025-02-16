use text_splitter::{ChunkConfig, TextSplitter};
use tokenizers::tokenizer::Tokenizer;

pub fn split(text: &str, tokenizer: &Tokenizer, max_tokens: usize) -> Vec<String> {
    // Load the Hugging Face tokenizer
    // let tokenizer = Tokenizer::from_file("all-MiniLM-L6-v2-tokenizer.json").unwrap();
    let splitter = TextSplitter::new(ChunkConfig::new(max_tokens).with_sizer(tokenizer));
    splitter.chunks(text).map(|s| s.to_string()).collect()
}
