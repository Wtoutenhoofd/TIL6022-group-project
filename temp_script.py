import pandas as pd
from datetime import datetime, timedelta

# Function to fix 24:00 timestamps
def fix_24_hour(datetime_str):
    """Convert 24:00 to 00:00 of the next day"""
    if ' 24:' in datetime_str:
        # Replace 24 with 00
        fixed_str = datetime_str.replace(' 24:', ' 00:')
        # Parse the datetime
        dt = pd.to_datetime(fixed_str, format='%Y%m%d %H:%M')
        # Add one day since 24:00 represents midnight of next day
        dt = dt + timedelta(days=1)
        return dt
    else:
        # Normal parsing for other times
        return pd.to_datetime(datetime_str, format='%Y%m%d %H:%M')

# Load the weather data
df = pd.read_csv('SAIL_Amsterdam_10min_Weather_2025-08-20_to_2025-08-24.csv')

# Convert DateTime column to proper datetime format
df['DateTime'] = df['DateTime'].apply(fix_24_hour)

print(f"Data loaded: {len(df)} records from {df['DateTime'].min()} to {df['DateTime'].max()}")

df.head()

# Create figure with subplots for each weather parameter
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

fig.suptitle('SAIL Amsterdam Weather Data (Aug 20-24, 2025)', fontsize=16, fontweight='bold')

# Set white background for cleaner look
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_facecolor('white')

# Plot 1: Temperature
axes[0].plot(df['DateTime'], df['Temperature_°C'], color='red', linewidth=1.5)
axes[0].set_ylabel('Temperature (°C)', fontsize=12)
axes[0].set_title('Temperature over Time')
axes[0].grid(True, alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Humidity
axes[1].plot(df['DateTime'], df['Humidity_%'], color='blue', linewidth=1.5)
axes[1].set_ylabel('Humidity (%)', fontsize=12)
axes[1].set_title('Humidity over Time')
axes[1].grid(True, alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)

# Plot 3: Rain
axes[2].plot(df['DateTime'], df['Rain_mm'], color='green', linewidth=1.5)
axes[2].set_xlabel('Date and Time', fontsize=12)
axes[2].set_ylabel('Rain (mm)', fontsize=12)
axes[2].set_title('Rainfall over Time')
axes[2].grid(True, alpha=0.3)
axes[2].tick_params(axis='x', rotation=45)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure
plt.savefig('weather_data_graphs.png', dpi=300, bbox_inches='tight')
print("Graph saved as 'weather_data_graphs.png'")

# Show the plot
plt.show()

# Summary statistics
print("=== Weather Data Summary ===")
print(f"\nTemperature (°C):")
print(f"  Min: {df['Temperature_°C'].min():.1f}°C")
print(f"  Max: {df['Temperature_°C'].max():.1f}°C")
print(f"  Mean: {df['Temperature_°C'].mean():.1f}°C")

print(f"\nHumidity (%):")
print(f"  Min: {df['Humidity_%'].min():.1f}%")
print(f"  Max: {df['Humidity_%'].max():.1f}%")
print(f"  Mean: {df['Humidity_%'].mean():.1f}%")

print(f"\nRainfall (mm):")
print(f"  Total: {df['Rain_mm'].sum():.2f} mm")
print(f"  Max (10-min): {df['Rain_mm'].max():.2f} mm")
print(f"  Rain events: {(df['Rain_mm'] > 0).sum()} (out of {len(df)} measurements)")

# Optional: Create a combined view on single plot
fig, ax1 = plt.subplots(figsize=(14, 6))

# Temperature on left y-axis
color = 'tab:red'
ax1.set_xlabel('Date and Time', fontsize=12)
ax1.set_ylabel('Temperature (°C)', color=color, fontsize=12)
ax1.plot(df['DateTime'], df['Temperature_°C'], color=color, linewidth=2, label='Temperature')
ax1.tick_params(axis='y', labelcolor=color)
ax1.tick_params(axis='x', rotation=45)

# Humidity on right y-axis
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Humidity (%)', color=color, fontsize=12)
ax2.plot(df['DateTime'], df['Humidity_%'], color=color, linewidth=2, alpha=0.7, label='Humidity')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Temperature and Humidity - Combined View', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()