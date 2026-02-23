"""
Fix CSV Data - Diagnose and repair city scores
"""

import pandas as pd
import numpy as np

print("="*70)
print("DIAGNOSING DATA FILES")
print("="*70)

# Check city_aspect_scores.csv
print("\n1. Checking city_aspect_scores.csv...")
try:
    df = pd.read_csv('data/city_aspect_scores.csv')
    print(f"   ✓ File found: {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Check if we have the aspect columns
    has_aspects = all(col in df.columns for col in ['comfort', 'cost', 'safety', 'social'])
    print(f"\n   Has all aspects: {has_aspects}")
    
    if not has_aspects:
        print("\n   ⚠ Missing aspect columns! Let me check what we have...")
        print(f"   Available columns: {df.columns.tolist()}")
        
except FileNotFoundError:
    print("   ✗ File not found!")
    df = None

# If the data structure is wrong, recreate it properly
if df is not None and 'city' in df.columns:
    print("\n2. Recreating final_city_scores.csv with correct data...")
    
    # Check if it's in pivot format or long format
    if 'aspect' in df.columns:
        print("   Data is in long format, pivoting...")
        # Pivot from long to wide format
        df_pivot = df.pivot(index='city', columns='aspect', values='sentiment_score')
        df_pivot = df_pivot.reset_index()
        
        # Rename columns
        df_pivot.columns.name = None
        
        # Ensure we have all required aspects
        for aspect in ['comfort', 'cost', 'safety', 'social']:
            if aspect not in df_pivot.columns:
                df_pivot[aspect] = 3.5
        
        output = df_pivot[['city', 'comfort', 'cost', 'safety', 'social']].copy()
    else:
        print("   Data is already in wide format...")
        # Ensure all columns exist
        for aspect in ['comfort', 'cost', 'safety', 'social']:
            if aspect not in df.columns:
                df[aspect] = 3.5
        
        output = df[['city', 'comfort', 'cost', 'safety', 'social']].copy()
    
    # Capitalize columns
    output.columns = ['city', 'Comfort', 'Cost', 'Safety', 'Social']
    
    # Convert to numeric and fill NaN
    for col in ['Comfort', 'Cost', 'Safety', 'Social']:
        output[col] = pd.to_numeric(output[col], errors='coerce')
        # Fill NaN with column mean
        col_mean = output[col].mean()
        if pd.isna(col_mean):
            col_mean = 3.5
        output[col] = output[col].fillna(col_mean).round(1)
    
    # Sort by city
    output = output.sort_values('city').reset_index(drop=True)
    
    # Save
    output.to_csv('data/final_city_scores.csv', index=False)
    
    print("\n" + "="*70)
    print("FIXED DATA:")
    print("="*70)
    print(output.to_string(index=False))
    
    print(f"\n✓ Saved to: data/final_city_scores.csv")
    
    # Statistics
    print("\n" + "="*70)
    print("STATISTICS:")
    print("="*70)
    print(f"Cities: {len(output)}")
    print(f"Comfort range: {output['Comfort'].min():.1f} - {output['Comfort'].max():.1f}")
    print(f"Cost range: {output['Cost'].min():.1f} - {output['Cost'].max():.1f}")
    print(f"Safety range: {output['Safety'].min():.1f} - {output['Safety'].max():.1f}")
    print(f"Social range: {output['Social'].min():.1f} - {output['Social'].max():.1f}")
    
else:
    print("\n✗ Cannot process data - check if sentiment_analysis.py ran successfully")

print("\n" + "="*70)
print("NEXT STEPS:")
print("="*70)
print("1. Refresh your browser (Ctrl+F5)")
print("2. Check if data loads correctly now")
print("3. If still showing 3.5, open browser console (F12) for errors")