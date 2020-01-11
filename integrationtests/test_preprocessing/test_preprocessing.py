import pandas as pd
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from src.data import load_data
from src.preprocessing import NormaliseTextColumnsTransformer


def test_preprocessing_pipeline():
    print("\ntest_preprocessing_pipeline")

    test_data_path = "integrationtests/test_data.csv"
    expected_output_path = "integrationtests/test_preprocessing/preprocessed.csv"

    df = load_data(test_data_path, ["domain", "class"])

    pipeline = Pipeline([
        (
            "preprocess",
            NormaliseTextColumnsTransformer(
                "normed", ["domain"],
            )
        ),
    ])

    pipeline_output = pipeline.transform(df)
    column_names = pipeline["preprocess"].get_feature_names()
    print("column_names", column_names)

    try:
        assert(column_names == ['class', 'domain_normed'])
    except AssertionError:
        pytest.fail("Didn't get the expected `get_feature_names` from pipeline")

    try:
        assert(isinstance(pipeline_output, np.ndarray))
    except AssertionError:
        pytest.fail("Didn't get expected type from pipeline")

    print(pipeline_output)

    result_df = pd.DataFrame(
        pipeline_output,
        columns=column_names
    )

    # need to do the `fillna` as pandas will replace empty strings will np.nan
    expected_df = pd.read_csv(expected_output_path).fillna('')

    try:
        assert(result_df.equals(expected_df))
    except AssertionError:
        pytest.fail("Data resulting from transformation did not match expected.")

    return pipeline