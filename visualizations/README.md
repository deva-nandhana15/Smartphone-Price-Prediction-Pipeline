# 📊 Visualizations - Quick Start

## Overview

This project includes **comprehensive visualization capabilities** to analyze your smartphone price prediction model. You'll generate:

- **8 Static Charts** (high-resolution PNG images)
- **3 Interactive Dashboards** (HTML files you can open in your browser)

---

## 🚀 How to Generate Visualizations

### Step 1: Install Dependencies

Open your terminal/command prompt and run:

**Windows (CMD)**:

```cmd
pip install matplotlib seaborn plotly kaleido
```

**Windows (PowerShell)**:

```powershell
pip install matplotlib seaborn plotly kaleido
```

**Linux/Mac**:

```bash
pip3 install matplotlib seaborn plotly kaleido
```

### Step 2: Run Generation Scripts

**Option A - Use the automated script:**

**Windows**:

```cmd
scripts\generate_visualizations.bat
```

**Linux/Mac**:

```bash
chmod +x scripts/generate_visualizations.sh
./scripts/generate_visualizations.sh
```

**Option B - Run manually:**

```bash
# Generate static charts
python src/visualization/visualize_results.py

# Generate interactive dashboards
python src/visualization/create_interactive_dashboard.py
```

### Step 3: View Your Visualizations

All outputs are saved in the `visualizations/` folder:

```
visualizations/
├── feature_importance.png          ← Open with image viewer
├── training_convergence.png        ← Open with image viewer
├── price_distribution.png          ← Open with image viewer
├── error_distribution.png          ← Open with image viewer
├── actual_vs_predicted.png         ← Open with image viewer
├── metrics_summary.png             ← Open with image viewer
├── feature_correlation.png         ← Open with image viewer
├── summary_dashboard.png           ← Open with image viewer
├── interactive_dashboard.html      ← Open in web browser! 🌐
├── 3d_price_analysis.html         ← Open in web browser! 🌐
└── metrics_gauges.html            ← Open in web browser! 🌐
```

---

## 📈 What You'll See

### Static Charts (PNG)

1. **Feature Importance**: Which features matter most for pricing?
2. **Training Convergence**: How did the model learn over 50 iterations?
3. **Price Distribution**: What's the typical smartphone price range?
4. **Error Distribution**: Where does the model make mistakes?
5. **Actual vs Predicted**: How accurate are the predictions?
6. **Metrics Summary**: RMSE, MAE, R² at a glance
7. **Feature Correlation**: How do features relate to each other?
8. **Summary Dashboard**: All-in-one overview

### Interactive Dashboards (HTML)

1. **Interactive Dashboard**: Zoom, pan, hover - explore your data!
2. **3D Price Analysis**: Rotate and explore Storage vs RAM vs Price
3. **Metrics Gauges**: Live gauge charts for performance metrics

**Pro Tip**: Double-click charts in HTML files to reset zoom!

---

## 🎨 Customization

Want to modify the visualizations? Edit these files:

- **Static charts**: `src/visualization/visualize_results.py`
- **Interactive dashboards**: `src/visualization/create_interactive_dashboard.py`

Change colors, add charts, modify layouts - it's all Python!

---

## 📊 Sample Visualizations

### Feature Importance

Shows which smartphone characteristics most influence price predictions:

- Screen Size: 34.82 (highest importance)
- Brand (Apple/Samsung): ~20-22
- Storage & RAM: Moderate importance
- Camera features: Lower but still significant

### Training Convergence

Visualizes model learning:

- Started at RMSE: $520
- Converged to RMSE: $405.56
- Smooth convergence indicates stable training

### Actual vs Predicted

Scatter plot comparing predictions to reality:

- Points near red line = accurate predictions
- Scattered points indicate R² of -0.0853
- Shows where model needs improvement

---

## 🐛 Troubleshooting

**Problem**: `ModuleNotFoundError: No module named 'matplotlib'`
**Solution**: Run `pip install matplotlib seaborn plotly kaleido`

**Problem**: HTML files won't open
**Solution**: Right-click → Open With → Choose your web browser (Chrome, Firefox, Edge)

**Problem**: Charts look blurry
**Solution**: Charts are 300 DPI - zoom in or use a better image viewer

**Problem**: Python not found
**Solution**:

- Windows: Install from python.org
- Mac: `brew install python3`
- Linux: `sudo apt-get install python3`

---

## 📚 Learn More

For detailed information about each visualization, see:

- **VISUALIZATION_GUIDE.md** - Complete documentation
- **README.md** - Project overview
- **FINAL_SUMMARY.md** - Results analysis

---

## 💡 Tips

1. **For Reports**: Use PNG images (professional quality)
2. **For Presentations**: Use interactive HTML (impressive!)
3. **For Analysis**: Start with summary_dashboard.png for overview
4. **For Deep Dive**: Open interactive_dashboard.html in browser

---

## ✨ Examples of Insights You'll Discover

- 📊 **Screen Size** is the #1 price predictor
- 📈 Model improved by **$114.44** during training
- 💰 Most phones cost between **$200-$600**
- ⚠️ **Higher-priced phones** have larger prediction errors
- 🔄 **50% of predictions** are within **$213** of actual price

---

**Ready to visualize your results?** Run the generation script and explore! 🚀

```cmd
scripts\generate_visualizations.bat
```
