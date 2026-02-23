"""
STEP 7: Generate Final Output Table
Creates final_city_scores.csv with format: city | Comfort | Cost | Safety | Social
"""

import pandas as pd
import numpy as np
import os

print("="*70)
print("STEP 7: GENERATING FINAL OUTPUT TABLE")
print("="*70)
print()

# Check if input file exists
if not os.path.exists('data/city_aspect_scores.csv'):
    print("❌ Error: data/city_aspect_scores.csv not found!")
    print("Please run sentiment_analysis.py first.")
    exit(1)

# Load city aspect scores
df = pd.read_csv('data/city_aspect_scores.csv')

print(f"Loaded data for {len(df)} cities")
print(f"Columns found: {df.columns.tolist()}\n")

# Check what columns we have
available_cols = df.columns.tolist()

# Ensure all required columns exist
required_aspects = ['comfort', 'cost', 'safety', 'social']
for aspect in required_aspects:
    if aspect not in available_cols:
        print(f"⚠ Warning: '{aspect}' column missing, adding with default value 3.5")
        df[aspect] = 3.5

# Select and format columns
output = df[['city', 'comfort', 'cost', 'safety', 'social']].copy()

# Capitalize column names to match required format
output.columns = ['city', 'Comfort', 'Cost', 'Safety', 'Social']

# Round all scores to 1 decimal place
output['Comfort'] = pd.to_numeric(output['Comfort'], errors='coerce').fillna(3.5).round(1)
output['Cost'] = pd.to_numeric(output['Cost'], errors='coerce').fillna(3.5).round(1)
output['Safety'] = pd.to_numeric(output['Safety'], errors='coerce').fillna(3.5).round(1)
output['Social'] = pd.to_numeric(output['Social'], errors='coerce').fillna(3.5).round(1)

# Sort alphabetically by city
output = output.sort_values('city').reset_index(drop=True)

# Display the table
print("="*70)
print("FINAL CITY ASPECT SCORES")
print("="*70)
print()
print(output.to_string(index=False))
print()

# Save to CSV
output.to_csv('data/final_city_scores.csv', index=False)
print(f"✅ Saved to: data/final_city_scores.csv")
print()

# Show statistics
print("="*70)
print("STATISTICS")
print("="*70)
print(f"Total cities: {len(output)}")
print(f"Average Safety: {output['Safety'].mean():.2f}")
print(f"Average Cost: {output['Cost'].mean():.2f}")
print(f"Average Social: {output['Social'].mean():.2f}")
print(f"Average Comfort: {output['Comfort'].mean():.2f}")
print()

print("="*70)
print("✅ ALL OUTPUTS GENERATED!")
print("="*70)
print()
print("📁 Files created:")
print("  ✅ data/final_city_scores.csv")
print()
print("🚀 Next: Open index.html in your browser")