import pandas as pd

def convert_date_format(date_str):
    month_abbr = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }

    try:
        # Check for "2013-01" format (year-month)
        if '-' in date_str and len(date_str.split('-')[0]) == 4:
            year, month = date_str.split('-')
            return f"{year}-{month}-01"

        # Handle "Jan-13" format
        elif '-' in date_str:
            month_part, year_part = date_str.split('-')

            # Process 2-digit years (assume 13 means 2013)
            full_year = f"20{year_part}" if len(year_part) == 2 else year_part

            # Get numeric month
            month_num = month_abbr[month_part]

            return f"{full_year}-{month_num}-01"

        else:
            print(f"Unrecognized date format: {date_str}")
            return date_str  # Return original for inspection

    except Exception as e:
        print(f"Failed to parse date: {date_str}, Error: {e}")
        return date_str  # Return original value for debugging


def process_csv_file(input_file, output_file):
    try:
        # Read CSV file
        df = pd.read_csv(input_file)

        # Verify date column exists
        if 'date' not in df.columns:
            print("Error: 'date' column not found in CSV file")
            return

        # Convert date formats
        print("Converting date formats...")
        df['date'] = df['date'].astype(str).apply(convert_date_format)

        # Save processed file
        df.to_csv(output_file, index=False)
        print(f"Processing complete! Results saved to: {output_file}")

        # Display sample output
        print("\nFirst 5 rows of converted data:")
        print(df.head())

    except Exception as e:
        print(f"Error processing file: {e}")


if __name__ == "__main__":
    # File paths
    input_csv = "final_monthly_summary_with_coords.csv"
    output_csv = "final_monthly_summary_with_coords_converted.csv"

    # Execute processing
    process_csv_file(input_csv, output_csv)