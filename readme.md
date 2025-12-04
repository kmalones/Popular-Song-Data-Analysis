# Popular Song Data Analysis

## Overview

This project analyzes Spotify 2024 data to predict whether a song will be a "Top Hit" (ranked in the top 100 all-time) using machine learning classification models.

## Dataset

- **Source**: `spotify_2024.csv`
- **Target Variable**: `Is Top Hit` (1 if All Time Rank ≤ 100, 0 otherwise)
- **Features**: 20+ audio and streaming metrics including Spotify streams, YouTube views, TikTok engagement, etc.

## Key Features

- Data cleaning and preprocessing (handles missing values, numeric conversion)
- Decision Tree Classifier with optimized hyperparameters (max_depth=4)
- Train-test split (75-25)
- Model evaluation metrics: accuracy, AUC-ROC, confusion matrix, classification report
