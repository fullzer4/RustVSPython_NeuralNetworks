pub mod data {

    use polars::prelude::{CsvReader, PolarsError};

    pub fn get_data() -> Result<CsvReader<'static, std::fs::File>, PolarsError> {

        let mut data = CsvReader::from_path("./train.csv");
        

        return data
    }
}
