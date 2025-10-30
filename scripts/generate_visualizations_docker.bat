@echo off
REM Generate visualizations using Docker containers

echo ============================================================
echo GENERATING VISUALIZATIONS IN DOCKER ENVIRONMENT
echo ============================================================

echo.
echo Installing visualization dependencies in spark-master...
docker exec -u root spark-master pip install matplotlib seaborn plotly kaleido --quiet

echo.
echo Copying visualization scripts to container...
docker cp src\visualization\visualize_results.py spark-master:/tmp/visualize_results.py
docker cp src\visualization\create_interactive_dashboard.py spark-master:/tmp/create_interactive_dashboard.py

echo.
echo Generating static visualizations...
docker exec spark-master python3 /tmp/visualize_results.py

echo.
echo Generating interactive dashboard...
docker exec spark-master python3 /tmp/create_interactive_dashboard.py

echo.
echo Copying generated visualizations back to host...
docker cp spark-master:/opt/spark/work-dir/visualizations ./visualizations

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
