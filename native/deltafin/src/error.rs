use std::error::Error;
use std::fmt::{self, Display, Formatter};

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DeltafinError {
    message: String,
}

impl DeltafinError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl Display for DeltafinError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        self.message.fmt(formatter)
    }
}

impl Error for DeltafinError {}

impl From<&str> for DeltafinError {
    fn from(message: &str) -> Self {
        Self::new(message)
    }
}

impl From<String> for DeltafinError {
    fn from(message: String) -> Self {
        Self::new(message)
    }
}

pub type Result<T> = std::result::Result<T, DeltafinError>;
