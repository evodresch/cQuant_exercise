# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:35:46 2025

@author: EvandroDresch
"""

# Import required packages for the analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Setting directory structure
HISTORICAL_PRICE_DATA_DIR = r"C:\Users\EvandroDresch\Downloads\cQuant\input\historicalPriceData\\"
SUPPLEMENTAL_MATERIALS_DATA_DIR = r"C:\Users\EvandroDresch\Downloads\cQuant\input\supplementalMaterials\\"
OUTPUT_DIR = r"C:\Users\EvandroDresch\Downloads\cQuant\output\\"

# Other general definitions
# Date formatter for the graphs later on
date_formatter = mdates.DateFormatter('%b-%y')

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
# BONUS 1 - MEAN PLOTS
# Generate two line plots that display the monthly average prices you computed
# in task 2 in cronological order (grouped_historical_data)

# Create a function to generate and save a simple line plot for average price
# of a data frame
def create_line_plot(df, output_path, data_type):
    
    fig, ax = plt.subplots()
    
    # Loop through the settlement points and add to plot
    for sp, group in df.groupby('SettlementPoint'):
        ax.plot(group['Date'], group['AveragePrice'], label=sp)
    
    # Labels and graph info
    ax.set_xlabel('Month')
    
    # Format dates to save space
    ax.xaxis.set_major_formatter(date_formatter)
    
    # Rotate the labels in the x axis
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Remaining labels
    ax.set_ylabel('Average Price')
    ax.set_title(f'Average Prices for {data_type}')
    ax.legend(title=data_type)
    
    # Save the plot
    plt.savefig(output_path)
    

# Get data for plot 1 (Settlement Hubs)
mask = grouped_historical_data['SettlementPoint'].str.startswith('HB')
plot1_data = grouped_historical_data[mask].copy()
plot1_data['Date'] = pd.to_datetime(plot1_data['Period'])

# Generate plot 1
output_plot1 = OUTPUT_DIR + "SettlementHubAveragePriceByMonth.png"
create_line_plot(plot1_data, output_plot1, "Settlement Hubs")

# Get data for plot 2 (Settlement Hubs)
mask = grouped_historical_data['SettlementPoint'].str.startswith('LZ')
plot1_data = grouped_historical_data[mask].copy()
plot1_data['Date'] = pd.to_datetime(plot1_data['Period'])

# Generate plot 1
output_plot1 = OUTPUT_DIR + "LoadZoneAveragePriceByMonth.png"
create_line_plot(plot1_data, output_plot1, "Load Zones")

# BONUS 1 ANSWER: The plots were created
print('BONUS TASK 1 ANSWER: Both plots were generated and saved to output')
print('------------------------------------------------------------------------')


###############################################################################
# BONUS 2 - VOLATILITY PLOTS
# Create plots that compare volatility across settlement hubs from task 4
# hub_price_vol

# Plot 1: Box plot of hourly volatility for each settlement hubs

# Group data according to the settlement hubs
grouped_price_vol_hubs = [group['HourlyVolatility'].values for _, group in \
                          hub_price_vol.groupby('SettlementPoint')]
# Get names of hubs for the box plots
labels = [hub for hub, _ in hub_price_vol.groupby('SettlementPoint')]

fig, ax = plt.subplots()
ax.boxplot(grouped_price_vol_hubs, labels=labels)

ax.set_title('Hourly Volatility of Price for each Settlement Hub \n 2016-2019')
ax.set_xlabel('Settlement Hub')
ax.set_ylabel('Hourly Volatility')

# Rotate the labels in the x axis
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
output_plot3 = OUTPUT_DIR + "BoxPlotHourlyVolSettlementHubs.png"
plt.savefig(output_plot3)

# PLot 2: Box plot of hourly volatility for each year
# Group data according to the settlement hubs
grouped_price_vol_hubs = [group['HourlyVolatility'].values for _, group in \
                          hub_price_vol.groupby('Year')]
# Get names of hubs for the box plots
labels = [hub for hub, _ in hub_price_vol.groupby('Year')]

fig, ax = plt.subplots()
ax.boxplot(grouped_price_vol_hubs, labels=labels)

ax.set_title('Hourly Volatility of Price for each year')
ax.set_xlabel('Year')
ax.set_ylabel('Hourly Volatility')

# Rotate the labels in the x axis
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
output_plot4 = OUTPUT_DIR + "BoxPlotHourlyVolYears.png"
plt.savefig(output_plot4)

# BONUS 2 ANSWER: Box plots were created to analyse the volatility of both 
# SettlementHubs throughout the whole period and also for the years for all
# hubs
print('BONUS TASK 2 ANSWER: Two boxplots were generated and saved to output')
print('Short analysis of box plots:')
print('Settlement Hub "HB_WEST" showed the highest volatility throughout the years')
print('Settlement Hub "HB_PAN" was added only in 2019 and with a very high volatility, significantly above HB_WEST')
print('2019 showed the highest volatility throughout the settlement hubs, with a significant outlier (HB PAN)')
print('------------------------------------------------------------------------')

###############################################################################
# BONUS 3 - HOURLY SHAPE PROFILE COMPUTATION
# Compute hourly shape profile for the settlement points

# Read the files from the formattedSpotHistory and add to a dictionary of dataframes
formatted_spot = {}
files_formatted = [f for f in os.listdir(OUTPUT_DIR + "/formattedSpotHistory/") if f.endswith('csv')]

for file in files_formatted:
    file_path = OUTPUT_DIR + "/formattedSpotHistory/" + file
    settlement_point = file.split('spot_')[1].split('.')[0]
    
    formatted_spot[settlement_point] = pd.read_csv(file_path, index_col='Variable', parse_dates=True)
    
    
for settlement_point in formatted_spot.keys():
    
    # Get dataframe from dictionary
    df = formatted_spot[settlement_point]
    
    # Create a result empty data frame
    hourly_shape_df = pd.DataFrame()
    
    # Add a variable for the day of the week
    df['Date'] = pd.to_datetime((df['Date']))
    df['DayOfWeek'] = df['Date'].dt.dayofweek + 1
    
    # Add a variable for the month
    df['Month'] = df['Date'].dt.month
    
    # Create a new variable for the combination day of week and year
    df['DayOfWeek_Month'] = df['DayOfWeek'].astype(str).str.zfill(2) + '_' \
        + df['Month'].astype(str).str.zfill(2)

    # Group the data frame by this new variable to calculate hourly shape profile
    grouped_df = df.groupby('DayOfWeek_Month')

    for g, group in grouped_df:
        
        # Get only data from the hours
        calc_df = group[[c for c in group.columns if c.startswith('X')]]
        
        # Calculate the average value for each hour for all dates
        hour_means = calc_df.mean(axis=0, numeric_only=True)
        
        # Normalize the 24 values to the mean of all hour means
        normalized_hourly_shape = pd.DataFrame(hour_means / hour_means.mean())
        # Convert to a data frame, transpose and add the DayOfWeek_Month as an index
        normalized_hourly_shape.columns = [g]
        normalized_hourly_shape = normalized_hourly_shape.T
        
        # Add to results data frame
        hourly_shape_df = pd.concat([hourly_shape_df, normalized_hourly_shape])
        
        # Add index name
        hourly_shape_df.index.name = 'DayOfWeek_Month'
        
        # Save the hourly shape profiles to csv
        output_path_bonus3 = OUTPUT_DIR + "hourlyShapeProfiles\\" + f"profile_{settlement_point}.csv"
        hourly_shape_df.to_csv(output_path_bonus3)
        
# BONUS 3 ANSWER: Hourly Shape Profiles generated
print('BONUS TASK 3 ANSWER: Hourly Shape Profiles were generated and saved to output')
print('------------------------------------------------------------------------')



















