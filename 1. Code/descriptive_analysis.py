from descriptive_stats import (
    calculate_default_stats,
    calculate_limit_stats,
    calculate_age_stats,
    calculate_sex_percentage,
    calculate_education_percentage,
    calculate_marriage_percentage
)

def run_descriptive_analysis(data):
    """Run all descriptive statistics analyses"""
    # Calculate statistics
    default_stats = calculate_default_stats(data)
    limit_stats = calculate_limit_stats(data)
    age_stats = calculate_age_stats(data)
    sex_percentage = calculate_sex_percentage(data)
    education_percentage = calculate_education_percentage(data)
    marriage_percentage = calculate_marriage_percentage(data)

    # Print results
    print("Default Payment Statistics:")
    print(f"Anzahl 1: {default_stats['count_1']}")
    print(f"Anzahl 0: {default_stats['count_0']}")
    print(f"In Prozent 1: {default_stats['percentage_1']}")
    print(f"In Prozent 0: {default_stats['percentage_0']}\n")
    
    print("Limit Balance Statistics:")
    print(f"Min: {limit_stats['min']}")
    print(f"Max: {limit_stats['max']}")
    print(f"Median: {limit_stats['median']}\n")
    
    print("Age Statistics:")
    print(f"Min: {age_stats['min']}")
    print(f"Max: {age_stats['max']}")
    print(f"Average: {age_stats['average']}\n")

    print("Anteil der Geschlechter:")
    print(sex_percentage)

    print("\nAnteil der Personen mit Universitärem Abschluss:")
    print(f"Percentage of educated individuals: {education_percentage}%")

    print("\nAnteil der Personen mit Heiratsstatus:")
    print(marriage_percentage)
