use polars::prelude::*;

pub fn getData() -> Result<(), Box<dyn std::error::Error>> {
    let csv_path = "caminho";

    let data_train = CsvReader::from_path(../../mnist/train.csv)?
        .infer_schema(None)
        .has_header(true)
        .finish()?;

    let data_test = CsvReader::from_path(../../mnist/train.csv)?
        .infer_schema(None)
        .has_header(true)
        .finish()?;

    println!("Data: {:?}", data);

    return [data_train, data_test]
}
