import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for publication-quality figures
plt.style.use('default')
sns.set_palette("husl")

# Model performance data (corrected and complete)
models = ['LSTM', 'Transformer', 'GRU', 'CNN-LSTM', 'Attention-LSTM', 'ARIMA', 'VAR']
rmse_values = [0.019, 0.025, 0.030, 0.027, 0.025, 0.060, 0.050]
mae_values = [0.015, 0.020, 0.024, 0.022, 0.020, 0.048, 0.039]
mape_values = [1.8, 2.3, 2.7, 2.5, 2.3, 5.2, 4.1]  # MAPE values in percentage
directional_accuracy = [65.6, 57.7, 56.9, 62.1, 64.4, 51.8, 52.6]

# Define colors for each model
colors = ['#2E8B57', '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#DDA0DD']

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Figure 11: Enhanced Model Performance Comparison\n(Including Transformers and State-of-the-Art Methods)', 
             fontsize=16, fontweight='bold', y=0.95)

# 1. RMSE Plot
bars1 = ax1.bar(models, rmse_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax1.set_title('Root Mean Squared Error (RMSE)', fontsize=14, fontweight='bold')
ax1.set_ylabel('RMSE Value', fontsize=12)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars1, rmse_values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 2. MAE Plot
bars2 = ax2.bar(models, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax2.set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
ax2.set_ylabel('MAE Value', fontsize=12)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars2, mae_values)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 3. MAPE Plot (FIXED)
bars3 = ax3.bar(models, mape_values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax3.set_title('Mean Absolute Percentage Error (MAPE)', fontsize=14, fontweight='bold')
ax3.set_ylabel('MAPE (%)', fontsize=12)
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars3, mape_values)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 4. Directional Accuracy Plot
bars4 = ax4.bar(models, directional_accuracy, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
ax4.set_title('Directional Accuracy', fontsize=14, fontweight='bold')
ax4.set_ylabel('Accuracy (%)', fontsize=12)
ax4.tick_params(axis='x', rotation=45)
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim(45, 75)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars4, directional_accuracy)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.90, hspace=0.3, wspace=0.3)

# Save the figure
plt.savefig('Figure_11_Enhanced_Performance_Comparison_Fixed.png', 
            dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

print("تم إصلاح الشكل 11 بنجاح - جميع المقاييس مكتملة الآن")

