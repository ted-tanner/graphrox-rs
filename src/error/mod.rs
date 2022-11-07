use std::fmt;

#[derive(Debug)]
pub enum GraphRoxError {
    NotFound(String),
    InvalidFormat(String),
}

#[allow(dead_code)]
impl GraphRoxError {
    fn into_inner(self) -> String {
        match self {
            GraphRoxError::NotFound(s) => s,
            GraphRoxError::InvalidFormat(s) => s,
        }
    }
}

impl fmt::Display for GraphRoxError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "GraphRoxError: {}",
            match self {
                GraphRoxError::NotFound(s) => s,
                GraphRoxError::InvalidFormat(s) => s,
            }
        )?;

        Ok(())
    }
}
