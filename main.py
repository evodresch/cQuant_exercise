# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:35:46 2025

@author: EvandroDresch
"""

# Import required packages for the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Setting directory structure
HISTORICAL_PRICE_DATA_DIR = r"C:\Users\EvandroDresch\Downloads\cQuant\input\historicalPriceData\\"
SUPPLEMENTAL_MATERIALS_DATA_DIR = r"C:\Users\EvandroDresch\Downloads\cQuant\input\supplementalMaterials\\"
OUTPUT_DIR = r"C:\Users\EvandroDresch\Downloads\cQuant\output\\"

###############################################################################
# TASK 1: Reading all the historical data and combining into a dataframe

# Creating a list of files containing historical data (csv)
files_list = sorted([file for file in os.listdir(HISTORICAL_PRICE_DATA_DIR) if file.endswith('csv')])

# Using pandas to read all files and adding to a dataframe
historical_price_data_df = pd.DataFrame()

for file in files_list:
    file_path = HISTORICAL_PRICE_DATA_DIR + file
    
    # Files are comma-separated, index_col is a date-time index
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Append to main data frame
    historical_price_data_df = pd.concat([historical_price_data_df, df])
    
# TASK 1 ANSWER: The data were loaded to the dataframe historical_price_data_df
print('TASK 1 ANSWER:')
print(historical_price_data_df.head())
print(historical_price_data_df.tail())
print(f'The dataframe has the following shape: {historical_price_data_df.shape}')
print('------------------------------------------------------------------------')

###############################################################################
# TASK 2: Compute average price for each settelement point and year-month in the
# dataset

# Group the dataframe by the Settlement Points and also by the year-month
grouped_historical_data = historical_price_data_df.groupby([pd.Grouper(freq='M'),
                                                            'SettlementPoint']).mean()
# Reset index
grouped_historical_data = grouped_historical_data.reset_index()
# Convert range index to datetime index
grouped_historical_data['Date'] = pd.to_datetime(grouped_historical_data['Date'])

# Rename column
grouped_historical_data = grouped_historical_data.rename(columns={'Price': 'AveragePrice',
                                                                      'Date': 'Period'})

# TASK 2 ANSWER: The average prices are located in the dataframe grouped_historical_data
print('TASK 2 ANSWER:')
print(grouped_historical_data.head())
print(grouped_historical_data.tail())
print('------------------------------------------------------------------------')

###############################################################################
# TASK 3: Write the computed monthly average prices to file as a CSV
# Columns: SettlementPoint, Year, Month, AveragePrice

# Copy the dataframe to generate output
monthly_average_prices = grouped_historical_data.copy()

# Generate necessary new columns
monthly_average_prices['Year'] = monthly_average_prices['Period'].dt.year
monthly_average_prices['Month'] = monthly_average_prices['Period'].dt.month

# Reorder columns
monthly_average_prices = monthly_average_prices[['SettlementPoint',
                                                 'Year',
                                                 'Month',
                                                 'AveragePrice']]

# Save to csv
output_path_task3 = OUTPUT_DIR + "AveragePriceByMonth.csv"
monthly_average_prices.to_csv(output_path_task3, index=False)

# TASK 3 ANSWER: The average prices were exported to AveragePriceByMonth.csv
print('TASK 3 ANSWER: AveragePriceByMonth.csv created')
print('------------------------------------------------------------------------')

###############################################################################
# TASK 4: Compute the hourly price volaticlity for each year and each settlement hub

# Create new dataframe containing only hubs
filter_mask = historical_price_data_df['SettlementPoint'].str[:2] == 'HB'
hub_price_data = historical_price_data_df[filter_mask]

# Filter prices <= 0
filter_mask = hub_price_data['Price'] > 0
hub_price_data = hub_price_data[filter_mask]
# Quick assert to check if data are all positive
assert (hub_price_data['Price'] > 0).all(), "Price data are not all positive."

# Formula to calculate price volatility
def calc_price_volatility(df):
    
    # Calculates the log returns of the column "Price" of a dataframe
    df['log_returns'] = np.log(df['Price'] / df['Price'].shift(1))
    
    # Calculates the price volatility
    price_volatility = df['log_returns'].std()
    
    # Return price volatility
    return price_volatility
    
# Group hub price data by settlement hub and year
hub_price_data_grouped = hub_price_data.groupby([pd.Grouper(freq='Y'),
                                                            'SettlementPoint']).apply(calc_price_volatility)

# Convert range index to datetime index
hub_price_vol = hub_price_data_grouped.reset_index()
hub_price_vol['Date'] = pd.to_datetime(hub_price_vol['Date']).dt.year


hub_price_vol.columns = ['Year', 'SettlementPoint', 'HourlyVolatility']

# TASK 4 ANSWER: The price_volatilty for the settlement hubs for each year were calculated

print('TASK 4 ANSWER: Price volatility for the settlement hubs aand years')
print(hub_price_vol.head())
print(hub_price_vol.tail())
print('------------------------------------------------------------------------')

###############################################################################
# TASK 5: Write the computed hourly volatilities for settlement hubs and years to csv

# Save to csv
output_path_task5 = OUTPUT_DIR + "HourlyVolatilityByYear.csv"
hub_price_vol.to_csv(output_path_task5, index=False)

# TASK 5 ANSWER: The average prices were exported to AveragePriceByMonth.csv
print('TASK 5 ANSWER: HourlyVolatilityByYear.csv created')
print('------------------------------------------------------------------------')

###############################################################################
# TASK 6: Determine which settlement hub showed the highes overall hourly 
# volatility for each year. Write to MaxVolatilityByYear.csv

# Get the index of the maximal values for each year
indexes_max = hub_price_vol.groupby('Year')['HourlyVolatility'].idxmax()

# Get the correspoding settlement hub and price
max_hub_price_vol = hub_price_vol.loc[indexes_max, ['Year', 'SettlementPoint', 'HourlyVolatility']]

# Save to csv
output_path_task6 = OUTPUT_DIR + "MaxVolatilityByYear.csv"
max_hub_price_vol.to_csv(output_path_task6, index=False)

# TASK 6 ANSWER: The average prices were exported to AveragePriceByMonth.csv
print('TASK 6 ANSWER: MaxVolatilityByYear.csv created')
print('------------------------------------------------------------------------')

###############################################################################
# TASK 7: Translate and format data from Task 1 into a format that can be
# consumed by the cQuant price simulation models
# Write the data to separate files for each settlement point

# Create a formula to format the grouped dataframes
def translate_format_data(df, settlement_point):
    # Translates and formats a dataframe from a long format to a wide format
    # according to the format of cQuant price simulation models
  
    # Create a column for each hour of the day, correct 0 to 1 (midnight)
    df['Hour'] = df.index.hour
    
    # Add "X" to hour column
    df['Hour'] = "X" + df['Hour'].astype(str).str.zfill(2)
    
    # Replace X0 by X24 (midnight)
    df['Hour'] = df['Hour'].str.replace('X00', 'X24')
    
    # Get the date only (no hour values)
    df['DateOnly'] = df.index.date
    
    # Create a wide dataframe
    df = df.pivot(columns='Hour', index='DateOnly', values='Price')
    df = df.reset_index()
    
    # Add settlement point to data frame
    df['Variable'] = settlement_point
    
    # Rename date column
    df = df.rename(columns={'DateOnly': 'Date'})
    
    # Sort columns
    df = df[sorted(df.columns)]
    df = df.set_index('Variable')
    
    # Remove 0 placeholder
    df.columns = [c.replace('X0', 'X') for c in df.columns]
    
    # Return translated dataframe
    return df

# Loop over the settlement points and create the output files
for settlement_point in historical_price_data_df['SettlementPoint'].unique():
    
    # Get the data
    mask = historical_price_data_df['SettlementPoint'] == settlement_point
    df = historical_price_data_df[mask].copy()
    
    # Translate the data for the settlement point
    translated_data = translate_format_data(df, settlement_point)
    
    # Save to csv
    output_path_task7 = OUTPUT_DIR + "formattedSpotHistory\\" + f"spot_{settlement_point}.csv"
    translated_data.to_csv(output_path_task7)

# TASK 7 ANSWER: The data were translated and the files created in the folder
print('TASK 7 ANSWER: Translated csv files saved in folder "formattedSpotHistory"')
print('------------------------------------------------------------------------')

###############################################################################
# BONUS - MEAN PLOTS














