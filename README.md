 #Inventory Forecasting & Par Level Recommendation System — README
 Overview

This project builds an end-to-end forecasting and inventory recommendation system for a hotel chain operating multiple bar locations. The goal is to prevent stockouts of fast-moving items and avoid overstocking slow movers, both of which increase operational cost and hurt guest satisfaction.

The system uses historical consumption data to:

Forecast future item demand

Recommend optimal par levels

Simulate day-to-day inventory usage

 Business Problem

Hotels with multiple bar outlets face recurring challenges:

High-demand items run out during peak hours

Slow-moving stock piles up, tying up capital & storage

Managers manually estimate stock levels, leading to errors

No standardized decision system across locations

These issues directly affect:

Guest satisfaction

Waste management

Cost control

Revenue consistency

A data-driven forecasting engine solves these operational inefficiencies.

Solution Approach
1. Data Exploration

Loaded historical consumption dataset

Cleaned missing values and checked item-level trends

Aggregated consumption by item and date

2. Demand Forecasting

The system uses a moving-average–based forecast because:

Data is simple, intermittent, and noisy

Many items have low or inconsistent daily consumption

Classical models like ARIMA need stable time series

The moving average is applied per item to create:

Short-term demand forecast

Expected daily usage per item

3. Par Level Calculation

Par levels are computed using:

Par Level = Forecasted Demand + Safety Stock


Where safety stock is calculated from:

Item demand variability

Lead time buffer

Usage deviations

This ensures bars maintain enough stock without overordering.

4. Inventory Simulation

The system simulates daily operations:

Starting stock

Predicted consumption

Remaining stock

Reorder triggers

This gives managers a realistic preview of how par levels perform in practice.

🛠️ Tech Stack

Python

pandas

numpy

matplotlib

scikit-learn (optional)

Jupyter Notebook
