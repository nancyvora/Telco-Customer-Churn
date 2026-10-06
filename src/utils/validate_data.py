import great_expectations as gx
import pandas as pd
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.

    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.

    """
    print("🔍 Starting data validation with Great Expectations...")

    # TotalCharges arrives from the CSV as text (blank for brand-new customers),
    # so check it as a number; blanks become nulls, which the range checks skip
    if "TotalCharges" in df.columns:
        df = df.assign(TotalCharges=pd.to_numeric(df["TotalCharges"], errors="coerce"))

    # Wrap the pandas DataFrame in a Great Expectations batch (in-memory, nothing written to disk)
    context = gx.get_context(mode="ephemeral")
    context.variables.progress_bars = {"globally": False}
    batch = (
        context.data_sources.add_pandas("telco")
        .add_dataframe_asset(name="telco_customers")
        .add_batch_definition_whole_dataframe("whole_dataframe")
        .get_batch(batch_parameters={"dataframe": df})
    )
    suite = gx.ExpectationSuite(name="telco_churn_suite")
    gxe = gx.expectations

    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")

    # Customer identifier must exist (required for business operations)
    suite.add_expectation(gxe.ExpectColumnToExist(column="customerID"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="customerID"))

    # Core demographic features
    suite.add_expectation(gxe.ExpectColumnToExist(column="gender"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="Partner"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="Dependents"))

    # Service features (critical for churn analysis)
    suite.add_expectation(gxe.ExpectColumnToExist(column="PhoneService"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="InternetService"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="Contract"))

    # Financial features (key churn predictors)
    suite.add_expectation(gxe.ExpectColumnToExist(column="tenure"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="MonthlyCharges"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="TotalCharges"))

    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")

    # Gender must be one of expected values (data integrity)
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="gender", value_set=["Male", "Female"]))

    # Yes/No fields must have valid values
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="Partner", value_set=["Yes", "No"]))
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="Dependents", value_set=["Yes", "No"]))
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column="PhoneService", value_set=["Yes", "No"]))

    # Contract types must be valid (business constraint)
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(
        column="Contract",
        value_set=["Month-to-month", "One year", "Two year"]
    ))

    # Internet service types (business constraint)
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(
        column="InternetService",
        value_set=["DSL", "Fiber optic", "No"]
    ))

    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")

    # Tenure must be non-negative (business logic - can't have negative tenure)
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="tenure", min_value=0))

    # Monthly charges must be positive (business logic - no free service)
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0))

    # Total charges should be non-negative (business logic)
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="TotalCharges", min_value=0))

    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")

    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120))

    # Monthly charges should be within reasonable business range
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="MonthlyCharges", min_value=0, max_value=200))

    # No missing values in critical numeric features
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="tenure"))
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="MonthlyCharges"))

    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")

    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    suite.add_expectation(gxe.ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95  # Allow 5% exceptions for edge cases
    ))

    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = batch.validate(suite)

    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results.results:
        if not r.success:
            expectation_type = r.expectation_config.type
            failed_expectations.append(expectation_type)

    # Print validation summary
    total_checks = len(results.results)
    passed_checks = sum(1 for r in results.results if r.success)
    failed_checks = total_checks - passed_checks

    if results.success:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")

    return results.success, failed_expectations
