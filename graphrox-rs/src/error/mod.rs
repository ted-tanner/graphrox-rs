use std::fmt;

/// An `enum` possessing possible GraphRox errors as variants.
#[derive(Debug)]
pub enum GraphRoxError {
    /// Indicates a representation of a graph is invalid. For example, a byte representation of
    /// a graph loaded from a file that has been corrupted.
    InvalidFormat(String),
    /// Indicates an input parameter is outside the supported range.
    InvalidInput(String),
    /// Indicates an operation would exceed supported memory or integer capacity.
    CapacityOverflow(String),
}

#[allow(dead_code)]
impl GraphRoxError {
    fn into_inner(self) -> String {
        match self {
            GraphRoxError::InvalidFormat(s) => s,
            GraphRoxError::InvalidInput(s) => s,
            GraphRoxError::CapacityOverflow(s) => s,
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
                GraphRoxError::InvalidInput(s) => s,
                GraphRoxError::CapacityOverflow(s) => s,
            }
        )?;

        Ok(())
    }
}
