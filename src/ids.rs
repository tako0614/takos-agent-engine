use std::fmt::{Display, Formatter};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

macro_rules! define_uuid_id {
    ($name:ident) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
        )]
        pub struct $name(pub Uuid);

        impl $name {
            pub fn new() -> Self {
                Self(Uuid::new_v4())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl FromStr for $name {
            type Err = uuid::Error;

            fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
                Uuid::parse_str(s).map(Self)
            }
        }
    };
}

define_uuid_id!(SessionId);
define_uuid_id!(LoopId);
define_uuid_id!(RawNodeId);
define_uuid_id!(AbstractNodeId);

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn new_ids_are_unique() {
        let a = SessionId::new();
        let b = SessionId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn default_produces_unique_ids() {
        let a = LoopId::default();
        let b = LoopId::default();
        assert_ne!(a, b);
    }

    #[test]
    fn display_matches_inner_uuid() {
        let id = RawNodeId::new();
        assert_eq!(id.to_string(), id.0.to_string());
    }

    #[test]
    fn from_str_roundtrip() {
        let original = AbstractNodeId::new();
        let text = original.to_string();
        let parsed: AbstractNodeId = text.parse().unwrap();
        assert_eq!(original, parsed);
    }

    #[test]
    fn from_str_invalid_returns_error() {
        let result = "not-a-uuid".parse::<SessionId>();
        assert!(result.is_err());
    }

    #[test]
    fn equality_and_clone() {
        let id = LoopId::new();
        let cloned = id;
        assert_eq!(id, cloned);
    }

    #[test]
    fn ids_are_hashable() {
        let id = RawNodeId::new();
        let mut set = HashSet::new();
        set.insert(id);
        set.insert(id);
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn ids_are_orderable() {
        let a = AbstractNodeId::new();
        let b = AbstractNodeId::new();
        // Just verify Ord is implemented and doesn't panic
        let _ = a.cmp(&b);
    }

    #[test]
    fn serialize_deserialize_json_roundtrip() {
        let id = SessionId::new();
        let json = serde_json::to_string(&id).unwrap();
        let deserialized: SessionId = serde_json::from_str(&json).unwrap();
        assert_eq!(id, deserialized);
    }

    #[test]
    fn serialize_format_is_uuid_string() {
        let id = RawNodeId::new();
        let json = serde_json::to_string(&id).unwrap();
        // JSON should be a quoted UUID string
        assert!(json.starts_with('"'));
        assert!(json.ends_with('"'));
        let inner = &json[1..json.len() - 1];
        assert_eq!(inner, id.0.to_string());
    }
}
