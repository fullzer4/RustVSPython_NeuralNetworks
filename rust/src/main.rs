//use ndarray
//use ndarray_rand
//use rand 
//ndarray_stats 
//polars
//serde
//serde_json
mod data;

use data::Data::get_data;

fn main() {
    let data = get_data();
    // [0] test dataset
    // [1] train dataset

    let data_train = &data[1];

    let data_test = &data[0];
}
