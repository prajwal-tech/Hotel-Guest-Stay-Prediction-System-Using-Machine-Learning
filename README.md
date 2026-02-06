Inventory Forecasting & Par Level Recommendation System
Overview

This project builds an end-to-end forecasting and inventory recommendation system for a hotel chain operating multiple bar locations. The goal is to reduce stockouts of fast-moving items and avoid overstocking slow-moving inventory, both of which increase operational cost and harm guest satisfaction.

Using historical consumption data, the system:

Forecasts item-level demand

Recommends optimal par levels

Simulates inventory usage and replenishment

Business Problem

Hotels with multiple bar outlets commonly struggle with:

Frequent stockouts during peak demand

Overstocking that ties up capital and storage space

Manual estimation of stock levels leading to inconsistent decisions

Lack of a standardized inventory planning system across locations

These issues negatively affect:

Guest satisfaction

Waste and spoilage

Cost control

Revenue stability

A data-driven forecasting engine helps reduce these inefficiencies.

Solution Approach
1. Data Exploration

Loaded and cleaned historical consumption data

Checked for missing values

Analyzed item-level trends and consumption patterns

Grouped data by item and date

2. Demand Forecasting

A moving average approach was used because:

Consumption data is noisy and intermittent

Many items have low or inconsistent daily usage

More complex forecasting models require stable patterns

This method produces:

Short-term demand estimates

Expected daily usage values

3. Par Level Calculation

Par levels are calculated using the formula:

Par Level = Forecasted Demand + Safety Stock

Safety stock is based on:

Variability in historical consumption

Lead time buffer

Unexpected demand spikes

This ensures stock availability without unnecessary overordering.

4. Inventory Simulation

A day-by-day simulation is performed to show:

Starting stock

Predicted usage

Remaining inventory

When reorders would be triggered

This helps validate that par levels are practical and sufficient.

Tech Stack

Python

pandas

numpy

matplotlib

Jupyter Notebook

How to Run

Place the dataset at:
Open and run the notebook:
Inventory_System_Notebook.ipynb

The system generates:

Demand forecasts

Par level recommendations

Simulation results

A CSV file with item-level recommendations

Outputs

Daily consumption forecast

Recommended par levels

Safety stock values

Inventory simulation summary

inventory_recommendations.csv
