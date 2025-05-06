#!/usr/bin/env python3
"""
Sample visualization code to demonstrate what evaluation results look like.
This can be used to generate example charts for the lab guide.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Create output directory
output_dir = f"results/sample_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(output_dir, exist_ok=True)

# Sample data - this simulates results from 10 test queries
categories = ["destination_recommendations", "itinerary_planning", "food_recommendations", 
              "family_travel", "travel_planning", "transportation", 
              "cultural_experiences", "safety", "sustainable_travel", "cultural_awareness"]

query_ids = [f"q{i+1}" for i in range(10)]

# Simulated results for Gemini and Gemma
np.random.seed(42)  # For reproducibility

gemini_response_times = np.random.uniform(5, 25, 10)
gemma_response_times = np.random.uniform(8, 30, 10)

gemini_quality_scores = np.random.uniform(0.6, 0.95, 10)
gemma_quality_scores = np.random.uniform(0.5, 0.85, 10)

gemini_specificity = np.random.uniform(0.4, 0.9, 10)
gemma_specificity = np.random.uniform(0.3, 0.8, 10)

# Create a sample dataframe
data = {
    'query_id': query_ids,
    'category': categories,
    'gemini_response_time': gemini_response_times,
    'gemma_response_time': gemma_response_times,
    'gemini_quality': gemini_quality_scores,
    'gemma_quality': gemma_quality_scores,
    'gemini_specificity': gemini_specificity,
    'gemma_specificity': gemma_specificity
}

df = pd.DataFrame(data)

# 1. Response Time Comparison
plt.figure(figsize=(14, 7))
bar_width = 0.35
index = np.arange(len(df))

plt.bar(index, df['gemini_response_time'], bar_width, label='Gemini', color='blue', alpha=0.7)
plt.bar(index + bar_width, df['gemma_response_time'], bar_width, label='Gemma', color='green', alpha=0.7)

plt.xlabel('Query')
plt.ylabel('Response Time (seconds)')
plt.title('Response Time Comparison: Gemini vs Gemma')
plt.xticks(index + bar_width/2, df['query_id'])
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/response_time_comparison.png", dpi=300)

# 2. Quality Score Comparison
plt.figure(figsize=(14, 7))
bar_width = 0.35
index = np.arange(len(df))

plt.bar(index, df['gemini_quality'], bar_width, label='Gemini', color='blue', alpha=0.7)
plt.bar(index + bar_width, df['gemma_quality'], bar_width, label='Gemma', color='green', alpha=0.7)

plt.xlabel('Query')
plt.ylabel('Quality Score (0-1)')
plt.title('Quality Score Comparison: Gemini vs Gemma')
plt.xticks(index + bar_width/2, df['query_id'])
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/quality_score_comparison.png", dpi=300)

# 3. Specificity Comparison
plt.figure(figsize=(14, 7))
bar_width = 0.35
index = np.arange(len(df))

plt.bar(index, df['gemini_specificity'], bar_width, label='Gemini', color='blue', alpha=0.7)
plt.bar(index + bar_width, df['gemma_specificity'], bar_width, label='Gemma', color='green', alpha=0.7)

plt.xlabel('Query')
plt.ylabel('Specificity Score (0-1)')
plt.title('Specificity Comparison: Gemini vs Gemma')
plt.xticks(index + bar_width/2, df['query_id'])
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/specificity_comparison.png", dpi=300)

# 4. Radar Chart for Category Performance
# Simulate category performance for travel-related metrics
categories = ['Attractions', 'Food', 'Accommodation', 'Transportation', 'Budget', 'Culture', 'Practical Info']

gemini_scores = np.random.uniform(0.6, 0.9, len(categories))
gemma_scores = np.random.uniform(0.5, 0.8, len(categories))

# Create radar chart
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)

# Compute angle for each category
angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()

# Close the loop
gemini_scores = np.append(gemini_scores, gemini_scores[0])
gemma_scores = np.append(gemma_scores, gemma_scores[0])
angles = angles + [angles[0]]
categories = categories + [categories[0]]

# Plot data
ax.plot(angles, gemini_scores, 'o-', linewidth=2, label='Gemini', color='blue')
ax.fill(angles, gemini_scores, alpha=0.25, color='blue')
ax.plot(angles, gemma_scores, 'o-', linewidth=2, label='Gemma', color='green')
ax.fill(angles, gemma_scores, alpha=0.25, color='green')

# Set category labels
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories[:-1])

# Set y-axis limit
ax.set_ylim(0, 1)

# Add title and legend
plt.title('Travel Category Performance: Gemini vs Gemma', size=15)
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig(f"{output_dir}/category_radar_chart.png", dpi=300)

# 5. Overall Comparison Summary
# Aggregate the scores
gemini_avg_response_time = np.mean(gemini_response_times)
gemma_avg_response_time = np.mean(gemma_response_times)
gemini_avg_quality = np.mean(gemini_quality_scores)
gemma_avg_quality = np.mean(gemma_quality_scores)
gemini_avg_specificity = np.mean(gemini_specificity)
gemma_avg_specificity = np.mean(gemma_specificity)

# Create summary bar chart
plt.figure(figsize=(12, 8))
metrics = ['Response Time\n(seconds)', 'Quality Score\n(0-1)', 'Specificity\n(0-1)']
gemini_scores = [gemini_avg_response_time, gemini_avg_quality, gemini_avg_specificity]
gemma_scores = [gemma_avg_response_time, gemma_avg_quality, gemma_avg_specificity]

x = np.arange(len(metrics))
bar_width = 0.35

plt.bar(x - bar_width/2, gemini_scores, bar_width, label='Gemini', color='blue', alpha=0.7)
plt.bar(x + bar_width/2, gemma_scores, bar_width, label='Gemma', color='green', alpha=0.7)

plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Overall Model Comparison Summary')
plt.xticks(x, metrics)
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/overall_comparison.png", dpi=300)

print(f"Sample visualizations created in {output_dir}") 