use std::cmp::Ordering;

pub struct VectorDB {
    vectors: Vec<(u32, Vec<f32>)>,
}

impl VectorDB {
    pub fn new() -> Self {
        VectorDB {
            vectors: Vec::new(),
        }
    }

    pub fn add_vector(&mut self, id: u32, vector: Vec<f32>) {
        self.vectors.push((id, vector));
    }

    pub fn get_vector(&self, id: u32) -> Option<&Vec<f32>> {
        self.vectors
            .iter()
            .find(|&&(vector_id, _)| vector_id == id)
            .map(|&(_, ref vector)| vector)
    }

    pub fn search(&self, query: &Vec<f32>, top_k: usize) -> Vec<(u32, f32)> {
        let mut results: Vec<(u32, f32)> = self
            .vectors
            .iter()
            .map(|&(id, ref vector)| (id, cosine_similarity(query, vector)))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        results.into_iter().take(top_k).collect()
    }
}

fn cosine_similarity(a: &Vec<f32>, b: &Vec<f32>) -> f32 {
    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot_product / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use crate::{cosine_similarity, VectorDB};

    #[test]
    fn test_add_and_get_vector() {
        let mut db = VectorDB::new();
        let vector = vec![1.0, 2.0, 3.0];
        db.add_vector(1, vector.clone());

        assert_eq!(db.get_vector(1), Some(&vector));
        assert_eq!(db.get_vector(2), None);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        assert_eq!(cosine_similarity(&a, &b), 1.0);
        assert_eq!(cosine_similarity(&a, &c), 0.0);
    }

    #[test]
    fn test_search() {
        let mut db = VectorDB::new();
        db.add_vector(1, vec![1.0, 0.0, 0.0]);
        db.add_vector(2, vec![0.0, 1.0, 0.0]);
        db.add_vector(3, vec![0.5, 0.5, 0.0]);

        let query = vec![1.0, 0.0, 0.0];
        let results = db.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 > 0.9);
        assert!(results[1].1 > 0.5);
    }
}
