use std::path::PathBuf;
use tokenizers::tokenizer::Tokenizer;

pub fn get_tokenizer() -> Tokenizer {
    let base_path = PathBuf::from("all-MiniLM-L6-v2");
    let tokenizer_path = base_path.join("tokenizer.json");
    Tokenizer::from_file(tokenizer_path).unwrap()
}
