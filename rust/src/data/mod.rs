pub mod Data {

    use polars::prelude::{CsvReader, PolarsError};

    pub fn get_data() -> [Result<CsvReader<'static, std::fs::File>, PolarsError>; 2] {

        let data_train = CsvReader::from_path("./train.csv");

        let data_test = CsvReader::from_path("./train.csv");

        return [data_train, data_test]
    }
}
