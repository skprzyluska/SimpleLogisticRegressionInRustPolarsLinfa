use polars::prelude::*;
use std::path::PathBuf;

use linfa::prelude::*;
use linfa_logistic::LogisticRegression;

fn quality_into_num(quality: &Column) -> Column {
    quality
        .i64()
        .unwrap()
        .into_iter()
        .map(|opt_qual: Option<i64>| opt_qual.map(|qual: i64| if qual > 5 { 1 } else { 0 }))
        .collect::<UInt8Chunked>()
        .into_column()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut df = CsvReadOptions::default()
        .with_infer_schema_length(Some(10000))
        .with_parse_options(
            CsvParseOptions::default()
                .with_separator(*";".as_bytes().to_owned().first().expect("empty")),
        )
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(project_path("Data/wine.csv")))?
        .finish()?;
    /*
    Do some data wrangling in Polars...
    */

    df.apply("quality", quality_into_num)
        .expect("Failed to convert column.");
    df = df
        .lazy()
        .drop_nulls(None)
        //.filter(all_horizontal([all().is_not_nan()])).expect("Failure getting rid of NaNs")
        .collect()?;
    println!("{:?}", df.head(Some(10)));

    let feature_array = df
        .select_by_range(0..(df.shape().1 - 2))
        .expect("Failed to select columns.")
        .to_ndarray::<Float32Type>(IndexOrder::C)?;
    let target_array = df
        .select(["quality"])
        .expect("No such column.")
        .to_ndarray::<UInt8Type>(IndexOrder::C)
        .expect("Failed making array.")
        .flatten()
        .to_owned()
        .mapv(|x| x != 0);

    // Replace with polars to ndarray
    let dataset = Dataset::new(feature_array, target_array);
    let (train, valid) = dataset.split_with_ratio(0.9);

    println!(
        "Fit Logistic Regression classifier with #{} training points",
        train.nsamples()
    );

    // fit a Logistic regression model with 150 max iterations
    let model = LogisticRegression::default()
        .max_iterations(150)
        .fit(&train)
        .unwrap();

    // predict and map targets
    let pred = model.predict(&valid);

    // create a confusion matrix
    let cm = pred.confusion_matrix(&valid).unwrap();

    // Print the confusion matrix, this will print a table with four entries. On the diagonal are
    // the number of true-positive and true-negative predictions, off the diagonal are
    // false-positive and false-negative
    println!("{cm:?}");

    // Calculate the accuracy and Matthew Correlation Coefficient (cross-correlation between
    // predicted and targets)
    println!("accuracy {}, MCC {}", cm.accuracy(), cm.mcc());

    Ok(())
}

pub fn project_path(relative_path: &str) -> PathBuf {
    let mut path = PathBuf::new();
    path.push(env!("CARGO_WORKSPACE_DIR"));
    path.push(relative_path);
    path
}
