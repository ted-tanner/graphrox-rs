use std::fmt;

/// An `enum` possessing possible GraphRox errors as variants.
#[derive(Debug)]
pub enum GraphRoxError {
    /// Indicates a representation of a graph is invalid. For example, a byte representation of
    /// a graph loaded from a file that has been corrupted.
    InvalidFormat(String),
}

#[allow(dead_code)]
impl GraphRoxError {
    fn into_inner(self) -> String {
        match self {
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
                GraphRoxError::InvalidFormat(s) => s,
            }
        )?;

        Ok(())
    }
}
