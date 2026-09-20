def first_10_days_missing(df, variable):
    """
    Check whether a behavioral variable is completely
    missing during the first 10 consecutive calendar days.

    If all first 10 calendar days are missing, the variable is
    excluded for that participant.

    Sleep variables are exempt from this rule.
    """

    check = df[
        ["Date", variable]
    ].copy()

    check = check.sort_values(
        "Date"
    )

    if check.empty:
        return False

    # Start from the participant's first recorded date
    start_date = check["Date"].min()

    # Define the first 10 consecutive calendar days
    first_10_dates = pd.date_range(
        start=start_date,
        periods=FIRST_10_DAYS,
        freq="D"
    )

    check = check.set_index("Date")

    # Reindex so missing calendar dates are explicitly represented
    first_10 = check.reindex(
        first_10_dates
    )

    # Variable must be missing for all first 10 calendar days
    return first_10[variable].isna().all()
