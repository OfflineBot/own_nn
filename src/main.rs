use rarray::{Array1, Array2};
use rarray_output::InputData;
use rarray_output::VecJson;

fn main() {
    let iter = 1_000;
    let lr = 0.01;
    let hidden_layer = 10;
    let print_level = 100;

    let data = InputData::<f32>::read_data("nn_data.json").unwrap();
    let input_vec = data.input;
    let output_vec = data.output;
    let input = Array2::from_vec(input_vec);
    let output = Array2::from_vec(output_vec);
    let new_data_vec = VecJson::read_vec2("new_data.json").unwrap();
    let new_data: Array2<f32> = Array2::from_vec(new_data_vec);

    let e_minus = 1e-3;
    let input_layer = input.size()[1];
    let output_layer = output.size()[1];

    input.replace_zero(e_minus);
    output.replace_zero(e_minus);
    new_data.replace_zero(e_minus);

    let x_mean = input.mean();
    let x_std = input.std();
    x_std.replace_zero(e_minus);

    let y_mean = output.mean();
    let y_std = output.std();
    y_std.replace_zero(e_minus);

    let x_norm = (input - x_mean) / x_std;
    let y_norm = (output - y_mean) / y_std;

    let mut w1: Array2<f32> = Array2::new(input_layer, hidden_layer);
    let mut b1: Array1<f32> = Array1::new(hidden_layer);
    let mut w2: Array2<f32> = Array2::new(hidden_layer, output_layer);
    let mut b2: Array1<f32> = Array1::new(output_layer);

    let (min, max) = (-1.0, 1.0);
    let seed = 32;

    w1.random_uniform_seed(min, max, seed);
    b1.random_uniform_seed(min, max, seed);
    w2.random_uniform_seed(min, max, seed);
    b2.random_uniform_seed(min, max, seed);

    for i in 0..=iter {

        let z1 = x_norm.dot(&w1) + &b1;
        let a1 = z1.relu();
        let z2 = a1.dot(&w2) + &b2;

        if i % print_level == 0 {
            println!("i: {:.2}", i as f32 / iter as f32 * 100.0);
        }

        let delta2 = z2 - y_norm;
        let delta1 = delta2.dot(&w2.t()) * z1.deriv_relu();

        w1 -= x_norm.t().dot(&delta1) * lr;
        b1 -= delta1.sum() * lr;
        w2 -= a1.t().dot(&delta2) * lr;
        b2 -= delta2.sum() * lr;
    }

    let new_norm = (new_data - x_mean) / x_std;
    let z1 = new_norm.dot(&w1) + &b1;
    let a1 = z1.relu();
    let z2 = a1.dot(&w2) + &b2;
    let output = z2 * y_std + y_mean;
    println!("output: {:?}", output);
}
