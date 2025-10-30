@echo off
REM Generate visualizations for Smartphone Price Prediction Pipeline

echo ============================================================
echo GENERATING VISUALIZATIONS
echo ============================================================

echo.
echo Installing visualization dependencies...
pip install matplotlib seaborn plotly kaleido --quiet

echo.
echo Generating static visualizations...
python src\visualization\visualize_results.py

echo.
echo Generating interactive dashboard...
python src\visualization\create_interactive_dashboard.py

echo.
echo ============================================================
echo VISUALIZATION GENERATION COMPLETE!
echo ============================================================
echo.
echo Generated files in visualizations\ folder:
echo   - Static PNG images (8 files)
echo   - Interactive HTML dashboards (3 files)
echo.
echo Open visualizations\interactive_dashboard.html in your browser!
echo.
pause
