use polars::prelude::*;

pub fn getData() -> Result<(), Box<dyn std::error::Error>> {
    let csv_path = "caminho";

    let data = CsvReader::from_path(csv_path)?
        .infer_schema(None)
        .has_header(true)
        .finish()?;

    println!("Data: {:?}", data);

    return data
}
