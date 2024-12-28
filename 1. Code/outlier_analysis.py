from descriptive_stats import (
    analyze_bill_amt_outliers,
    analyze_pay_amt_outliers,
    analyze_bill_amt_outliers_z,
    analyze_pay_amt_outliers_z
)

def run_outlier_analysis(data):
    """Run all outlier analyses"""
    # IQR method
    bill_amt_outliers = analyze_bill_amt_outliers(data)
    pay_amt_outliers = analyze_pay_amt_outliers(data)
    
    print("\nBILL_AMT Outlier Analysis:")
    for column, info in bill_amt_outliers.items():
        print(f"{column}: {info['count']} Ausreisser")
        print(f"  IQR: {info['IQR']:.2f}")
        print(f"  Bounds: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")

    print("\nPAY_AMT Outlier Analysis:")
    for column, info in pay_amt_outliers.items():
        print(f"{column}: {info['count']} Ausreisser")
        print(f"  IQR: {info['IQR']:.2f}")
        print(f"  Bounds: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")

    # Z-Score method
    bill_amt_outliers_z = analyze_bill_amt_outliers_z(data)
    pay_amt_outliers_z = analyze_pay_amt_outliers_z(data)

    print("\nBILL_AMT Outlier Analysis (Z-Score):")
    for column, info in bill_amt_outliers_z.items():
        print(f"{column}: {info['count']} Ausreisser gefunden (Schwellenwert: {info['threshold']})")

    print("\nPAY_AMT Outlier Analysis (Z-Score):")
    for column, info in pay_amt_outliers_z.items():
        print(f"{column}: {info['count']} Ausreisser gefunden (Schwellenwert: {info['threshold']})")
