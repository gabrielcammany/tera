use std::vec;

use vector_db::VectorDB;

fn main() {
    let mut db = VectorDB::new();

    // load test to 1M vectors of 1000 dimensions
    for i in 0..92000 {
        let vector = vec![i as f32; 1024];
        db.add_vector(i, vector);
    }

    if let Some(vector) = db.get_vector(1) {
        println!("Vector with ID 1: {:?}", vector);
    }

    let query = vec![0.1; 1024];
    let start = std::time::Instant::now();
    let results = db.search(&query, 2);
    // println!("Search results: {:?}", results);
    println!("Search time: {:?}", start.elapsed());
}
