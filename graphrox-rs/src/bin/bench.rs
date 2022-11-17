use std::num::Wrapping;
use std::time::Instant;

use graphrox::matrix::{CsrSquareMatrix, MatrixRepresentation};
use graphrox::{Graph, GraphRepresentation};

fn main() {
    let mut graph = Graph::new_undirected();

    println!("{:#?}", graph);
    println!();

    let edges = [4, 2, 1, 0, 9];
    graph.add_vertex(10, Some(&edges));
    graph.add_edge(10, 1);
    graph.delete_edge(10, 1);
    graph.add_vertex(12, None);

    println!("{:#?}", graph);
    println!();

    println!("{:#?}", graph.clone());
    println!();

    println!("{}", graph.matrix_representation_string());
    println!();

    let graph = <Graph as Into<Vec<u8>>>::into(graph);
    let mut graph = Graph::try_from(&graph[..]).unwrap();

    println!("{:#?}", graph);
    println!();

    println!("{}", graph.matrix_representation_string());
    println!();

    graph.add_edge(0, 0);
    graph.add_edge(0, 1);
    graph.add_edge(1, 2);
    graph.add_edge(3, 2);
    graph.add_edge(1, 2);
    graph.add_edge(2, 2);

    graph.add_edge(4, 5);
    graph.add_edge(5, 6);
    graph.add_edge(4, 6);
    graph.add_edge(5, 5);

    println!("{}", graph.matrix_representation_string());
    println!();

    let mut matrix = CsrSquareMatrix::new();

    matrix.set_entry(106531u64, 6, 5);
    matrix.set_entry(4, 2, 3);
    matrix.set_entry(10, 4, 1);
    matrix.set_entry(10, 0, 3);
    matrix.set_entry(1, 7, 2);
    matrix.set_entry(12, 5, 0);

    println!("{}", matrix.to_string());
    println!();

    let avg_pool_matrix = graph.find_avg_pool_matrix(4);

    println!("{}", avg_pool_matrix.to_string());
    println!();

    let approx_graph = graph.approximate(4, 0.3);

    println!("{}", approx_graph.matrix_representation_string());
    println!();

    let mut graph = Graph::new_undirected();

    const MAX_VERTEX_ID: u64 = 200000;

    for i in 0..(MAX_VERTEX_ID / 4) {
        let num1 = bad_random(i ^ 7) % 10000 + 10;
        let num2 = bad_random(i ^ 37) % 10000 + 10;

        graph.add_edge(num1, num2);
    }

    for i in (MAX_VERTEX_ID / 4)..(MAX_VERTEX_ID / 2) {
        let num1 = bad_random(i ^ 7) % 20000 + 20000;
        let num2 = bad_random(i ^ 37) % 20000 + 40000;

        graph.add_edge(num1, num2);
    }

    for i in (MAX_VERTEX_ID / 2)..(3 * MAX_VERTEX_ID / 4) {
        let num1 = bad_random(i ^ 7) % 1000 + 140000;
        let num2 = bad_random(i ^ 37) % 1000 + 70000;

        graph.add_edge(num1, num2);
    }

    for i in (3 * MAX_VERTEX_ID / 4)..MAX_VERTEX_ID {
        let num1 = bad_random(i ^ 7) % 5000 + 30000;
        let num2 = bad_random(i ^ 37) % 5000 + 39000;

        graph.add_edge(num1, num2);
    }

    println!("Beginning test");

    let before = Instant::now();
    for _ in 0..2 {
        graph.find_avg_pool_matrix(8);
    }

    println!(
        "Finding average pool matrix twice took {}ms",
        before.elapsed().as_millis()
    );

    let before = Instant::now();
    for i in 0..100000000 {
        graph.does_edge_exist(i, i);
    }

    println!(
        "Getting 100,000,000 edges took {}ms",
        before.elapsed().as_millis()
    );
}

// This is adapted from
// https://www.javamex.com/tutorials/random_numbers/xorshift.shtml#.VlcaYzKwEV8
fn bad_random(seed: u64) -> u64 {
    let mut seed = Wrapping(seed);
    seed ^= seed << 21;
    seed ^= seed >> 35;
    seed ^= seed << 4;

    seed.0
}
