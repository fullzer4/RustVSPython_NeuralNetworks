pub mod Data {

    use polars::prelude::*;

    pub fn getData() {
        let csv_path = "caminho";

        let data = CsvReader::from_path("./train.csv")?
            .infer_schema(None)
            .has_header(true)
            .finish(); 

        println!("Data: {:?}", data);

        return data
    }
}
