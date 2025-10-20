import pandas as pd

# Read the CSV file
df = pd.read_csv('SAIL_Amsterdam_10min_Weather_2025-08-20_to_2025-08-24.csv')

# Fix the DateTime column by replacing '24:' with '00:' and incrementing the date
def fix_datetime(row):
    dt_str = row['DateTime']
    # Check if it contains '24:' hour
    if ' 24:' in dt_str:
        # Split into date and time parts
        date_part, time_part = dt_str.split(' ')
        # Replace 24: with 00:
        time_part = time_part.replace('24:', '00:')
        # Increment the date by 1 day
        year = int(date_part[:4])
        month = int(date_part[4:6])
        day = int(date_part[6:8])
        
        # Simple date increment logic
        if month == 8 and day == 20:
            new_date = '20250821'
        elif month == 8 and day == 21:
            new_date = '20250822'
        elif month == 8 and day == 22:
            new_date = '20250823'
        elif month == 8 and day == 23:
            new_date = '20250824'
        elif month == 8 and day == 24:
            new_date = '20250825'
        else:
            new_date = date_part
        
        return new_date + ' ' + time_part
    return dt_str

# Apply the fix
df['DateTime'] = df.apply(fix_datetime, axis=1)

# Save the corrected file
df.to_csv('SAIL_Amsterdam_10min_Weather_2025-08-20_to_2025-08-24.csv', index=False)

print("DateTime column has been fixed successfully!")
print(f"Total records: {len(df)}")
