# Visualization Guide - Smartphone Price Prediction

## 📊 Overview

This project includes comprehensive visualization capabilities to analyze model performance, data distributions, and prediction accuracy. Visualizations are available in two formats:

1. **Static Charts** (PNG) - High-resolution publication-ready images (8 charts)
2. **Interactive Dashboards** (HTML) - Web-based interactive visualizations (3 dashboards)

**Total Output**: 11 professional visualizations ready for reports, presentations, and analysis.

---

## 🎉 What's Included

### Visualization Scripts (2 files)

- `src/visualization/visualize_results.py` - Generates 8 static PNG charts (300 DPI)
- `src/visualization/create_interactive_dashboard.py` - Generates 3 interactive HTML dashboards

### Automation Scripts (2 files)

- `scripts/generate_visualizations.bat` - Windows one-click generation
- `scripts/generate_visualizations.sh` - Linux/Mac one-click generation

### Libraries Added

- **matplotlib** 3.6.0 - Static plot generation
- **seaborn** 0.12.0 - Statistical visualizations
- **plotly** 5.11.0 - Interactive HTML dashboards
- **kaleido** 0.2.1 - Static image export

---

## 🚀 Quick Start

### Generate All Visualizations

**Windows**:

```cmd
scripts\generate_visualizations.bat
```

**Linux/Mac**:

```bash
chmod +x scripts/generate_visualizations.sh
./scripts/generate_visualizations.sh
```

### View Results

All visualizations are saved in the `visualizations/` folder:

- **PNG files**: Open with any image viewer (8 static charts at 300 DPI)
- **HTML files**: Open in your web browser - Chrome, Firefox, Edge recommended (3 interactive dashboards)

### Manual Execution (Alternative)

If automation scripts don't work, run manually:

```bash
# Install dependencies (one time)
pip install matplotlib seaborn plotly kaleido

# Generate static charts
python src/visualization/visualize_results.py

# Generate interactive dashboards
python src/visualization/create_interactive_dashboard.py
```

---

## 📊 Complete Visualization Reference

### Static Charts (8 PNG Images)

| #   | Chart Name                 | Purpose                | Key Insight                |
| --- | -------------------------- | ---------------------- | -------------------------- |
| 1   | `feature_importance.png`   | Top 15 features        | Screen Size #1 (34.82)     |
| 2   | `training_convergence.png` | RMSE over iterations   | 22% improvement            |
| 3   | `price_distribution.png`   | Market distribution    | Multi-modal pricing        |
| 4   | `error_distribution.png`   | 4-panel error analysis | Errors increase with price |
| 5   | `actual_vs_predicted.png`  | Scatter with trend     | R² = -0.0853               |
| 6   | `metrics_summary.png`      | RMSE/MAE/R² gauges     | Quick performance view     |
| 7   | `feature_correlation.png`  | Correlation heatmap    | RAM-Storage: 0.72          |
| 8   | `summary_dashboard.png`    | 6-panel overview       | One-page summary           |

### Interactive Dashboards (3 HTML Files)

| #   | Dashboard Name               | Features                | Use Case               |
| --- | ---------------------------- | ----------------------- | ---------------------- |
| 1   | `interactive_dashboard.html` | 6 zoomable charts       | Data exploration       |
| 2   | `3d_price_analysis.html`     | Rotatable 3D scatter    | Multi-feature analysis |
| 3   | `metrics_gauges.html`        | Live performance gauges | Executive dashboards   |

---

## 📈 Static Visualizations (8 Charts)

### 1. Feature Importance (`feature_importance.png`)

**Description**: Horizontal bar chart showing the top 15 most influential features for price prediction.

**Key Insights**:

- Screen Size has the highest importance (34.82)
- Brand features (Apple, Samsung) significantly impact predictions
- Storage and RAM are crucial hardware factors

**Use Case**: Understand which smartphone characteristics most influence pricing decisions.

---

### 2. Training Convergence (`training_convergence.png`)

**Description**: Line plot showing RMSE decrease over 50 training iterations.

**Key Insights**:

- Initial RMSE: $520.00
- Final Training RMSE: $405.56
- Convergence achieved after ~40 iterations
- Test RMSE: $410.04 (marked with red dashed line)

**Use Case**: Verify model training stability and determine if more iterations are needed.

---

### 3. Price Distribution (`price_distribution.png`)

**Description**: Dual plot with histogram and box plot of smartphone prices.

**Key Insights**:

- Price range: $50 - $5,000
- Mean price: ~$380
- Median price: ~$300
- Multi-modal distribution (budget, mid-range, premium, flagship)

**Use Case**: Understand market price distribution and identify target segments.

---

### 4. Error Distribution (`error_distribution.png`)

**Description**: 4-panel visualization analyzing prediction errors.

**Panels**:

1. **Error Histogram**: Distribution of prediction errors (centered around $0)
2. **Absolute Error**: Distribution of prediction magnitudes
3. **Error by Price Range**: Average error for different price segments
4. **Cumulative Distribution**: Percentage of predictions within error thresholds

**Key Insights**:

- Errors normally distributed around $0
- 50% of predictions within $213 of actual price
- Higher-priced phones have larger absolute errors
- Error increases linearly with price range

**Use Case**: Identify where model performs well/poorly and detect systematic biases.

---

### 5. Actual vs Predicted (`actual_vs_predicted.png`)

**Description**: Scatter plot comparing actual prices to model predictions.

**Key Insights**:

- Red dashed line = perfect prediction
- Blue trend line shows actual model performance
- Points scattered widely indicate poor R² score
- Some clustering visible around popular price points

**Use Case**: Visual assessment of prediction accuracy and identify outliers.

---

### 6. Metrics Summary (`metrics_summary.png`)

**Description**: Three gauge charts displaying key performance metrics.

**Metrics**:

- **RMSE Gauge**: $410.04 (yellow zone = moderate error)
- **MAE Gauge**: $212.74 (green zone = acceptable)
- **R² Gauge**: -0.0853 (red zone = needs improvement)

**Use Case**: Quick visual summary of model performance at a glance.

---

### 7. Feature Correlation (`feature_correlation.png`)

**Description**: Heatmap showing correlations between features.

**Key Insights**:

- Strong correlation between RAM and Storage (0.72)
- Camera and Camera Score highly correlated (0.85)
- Price moderately correlated with Storage (0.65) and RAM (0.58)
- Screen size weakly correlated with other features

**Use Case**: Identify multicollinearity issues and feature dependencies.

---

### 8. Summary Dashboard (`summary_dashboard.png`)

**Description**: Comprehensive single-page dashboard with 6 panels.

**Panels**:

1. Key Metrics (text box with all statistics)
2. Training Convergence (line plot)
3. Price Distribution (histogram)
4. Error Distribution (histogram)
5. Top Features (horizontal bar chart)
6. Actual vs Predicted (scatter plot)

**Use Case**: One-page overview for presentations and reports.

---

## 🎨 Interactive Visualizations (3 Dashboards)

### 1. Interactive Dashboard (`interactive_dashboard.html`)

**Description**: Multi-panel interactive dashboard with zoom, pan, and hover capabilities.

**Features**:

- **Price Distribution**: Interactive histogram with hover tooltips
- **Actual vs Predicted**: Clickable scatter plot with zoom
- **Error Distribution**: Dynamic histogram
- **Feature Importance**: Sortable horizontal bar chart
- **Training Convergence**: Interactive line plot with data points
- **Error by Price Range**: Colorful bar chart

**Interactions**:

- Hover over any point to see exact values
- Click and drag to zoom into specific regions
- Double-click to reset zoom
- Use toolbar to save images or toggle traces

**Use Case**: Exploratory data analysis and interactive presentations.

---

### 2. 3D Price Analysis (`3d_price_analysis.html`)

**Description**: 3D scatter plot showing relationship between Storage, RAM, and Price.

**Features**:

- Rotate visualization by clicking and dragging
- Zoom in/out with scroll wheel
- Color gradient represents price levels
- Hover to see exact values for each data point

**Key Insights**:

- Clear upward trend as storage increases
- RAM has moderate positive effect
- Some outliers visible in 3D space

**Use Case**: Understand multi-dimensional relationships between features.

---

### 3. Metrics Gauges (`metrics_gauges.html`)

**Description**: Three interactive gauge charts for RMSE, MAE, and R².

**Features**:

- **RMSE Gauge**: Range 0-1000, target zones colored
- **MAE Gauge**: Range 0-500, performance bands
- **R² Gauge**: Range -1 to 1, baseline marker at 0

**Interactions**:

- Hover to see exact metric values
- Color-coded zones indicate performance levels
- Delta values show improvement needed

**Use Case**: Executive dashboards and performance monitoring.

---

## 📊 Visualization Technologies

### Libraries Used

| Library        | Version | Purpose                         |
| -------------- | ------- | ------------------------------- |
| **Matplotlib** | 3.6.0   | Static plot generation          |
| **Seaborn**    | 0.12.0  | Statistical visualizations      |
| **Plotly**     | 5.11.0  | Interactive HTML dashboards     |
| **Kaleido**    | 0.2.1   | Static image export from Plotly |
| **NumPy**      | 1.23.3  | Numerical computations          |
| **Pandas**     | 1.5.0   | Data manipulation               |

### Design Principles

1. **Color Palette**: Carefully selected colors for colorblind accessibility
2. **High Resolution**: All PNG images exported at 300 DPI
3. **Responsive Layout**: Interactive dashboards adapt to screen size
4. **Consistent Style**: Unified theme across all visualizations
5. **Annotations**: Key insights highlighted with text and markers

---

## 🎯 Interpretation Guide

### Understanding RMSE ($410.04)

- **Meaning**: On average, predictions deviate by $410 from actual prices
- **Context**: For a dataset with prices $50-$5,000, this represents ~8-16% error
- **Improvement**: Reduce RMSE by adding features or using ensemble methods

### Understanding MAE ($212.74)

- **Meaning**: Median absolute prediction error is $213
- **Context**: More robust to outliers than RMSE
- **Interpretation**: Half of predictions are within $213 of actual price

### Understanding R² (-0.0853)

- **Meaning**: Model performs worse than simply predicting the mean price
- **Negative Value**: Indicates fundamental model limitations
- **Action**: Consider non-linear models, more features, or different algorithm

---

## 📁 File Organization

```
visualizations/
├── feature_importance.png          # Top features bar chart
├── training_convergence.png        # RMSE over iterations
├── price_distribution.png          # Price histogram + box plot
├── error_distribution.png          # 4-panel error analysis
├── actual_vs_predicted.png         # Scatter plot with trend
├── metrics_summary.png             # 3 gauge charts
├── feature_correlation.png         # Correlation heatmap
├── summary_dashboard.png           # 6-panel overview
├── interactive_dashboard.html      # Interactive multi-panel
├── 3d_price_analysis.html         # 3D scatter visualization
└── metrics_gauges.html            # Interactive gauges
```

---

## 🛠️ Customization

### Modify Visualization Scripts

**Static Charts**: Edit `src/visualization/visualize_results.py`

```python
# Example: Change color scheme
sns.set_palette("husl")  # Line 13

# Example: Adjust figure size
plt.rcParams['figure.figsize'] = (14, 10)  # Line 14
```

**Interactive Dashboards**: Edit `src/visualization/create_interactive_dashboard.py`

```python
# Example: Change plot template
fig.update_layout(template='plotly_dark')  # Line 143

# Example: Modify color scale
marker=dict(colorscale='Plasma')  # Line 220
```

### Add New Visualizations

1. Create function in visualization script
2. Add function call in `main()`
3. Run generation script to produce output

---

## 💡 Tips for Presentations

### For Academic Reports

- Use **static PNG images** for consistency
- Include `summary_dashboard.png` for overview
- Reference `feature_importance.png` for model insights
- Add `error_distribution.png` for detailed analysis

### For Business Presentations

- Use **interactive HTML dashboards** for engagement
- Start with `metrics_gauges.html` for impact
- Demonstrate `interactive_dashboard.html` for exploration
- Show `3d_price_analysis.html` for impressive visuals

### For Technical Documentation

- Include all **static charts** in appendix
- Reference specific visualizations in analysis sections
- Use `feature_correlation.png` for technical discussions
- Include `training_convergence.png` for model tuning details

---

## 🚀 Advanced Usage

### Export Plotly to Static Images

If you need PNG versions of interactive dashboards:

```python
import plotly.io as pio

# Configure kaleido
pio.kaleido.scope.default_format = "png"
pio.kaleido.scope.default_width = 1920
pio.kaleido.scope.default_height = 1080

# Export
fig.write_image("output.png", format="png", scale=2)
```

### Batch Processing

Generate visualizations for multiple model runs:

```bash
# Run visualization script multiple times with different model files
for model in model_v1 model_v2 model_v3; do
    python src/visualization/visualize_results.py --model $model
done
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'matplotlib'"

**Solution**: Install dependencies

```bash
pip install -r requirements.txt
```

### Issue: Plotly charts not rendering in browser

**Solution**: Ensure JavaScript is enabled and use modern browser (Chrome/Firefox)

### Issue: Low-quality PNG images

**Solution**: Check DPI setting in script (should be 300)

```python
plt.savefig(path, dpi=300, bbox_inches='tight')
```

### Issue: Kaleido not found for static export

**Solution**: Install kaleido separately

```bash
pip install kaleido==0.2.1
```

---

## 📚 Additional Resources

### Documentation Links

- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [Plotly Documentation](https://plotly.com/python/)

### Color Palette References

- [ColorBrewer](https://colorbrewer2.org/) - Colorblind-safe palettes
- [Coolors](https://coolors.co/) - Color scheme generator

### Best Practices

- [Data Visualization Guide](https://clauswilke.com/dataviz/)
- [Matplotlib Style Sheets](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html)

---

## 📊 Visualization Checklist

Before finalizing visualizations for reports/presentations:

- [ ] All charts have clear titles
- [ ] Axes are properly labeled with units
- [ ] Color schemes are colorblind-friendly
- [ ] Text is readable at intended display size
- [ ] Legends are clear and positioned appropriately
- [ ] Grid lines enhance (not distract from) data
- [ ] File names are descriptive and organized
- [ ] High resolution (300 DPI for print, 72 DPI for web)
- [ ] Consistent styling across all visualizations
- [ ] Key insights are annotated or highlighted

---

**Generated**: October 30, 2025  
**Version**: 1.0  
**Author**: Big Data IA3 Project Team
