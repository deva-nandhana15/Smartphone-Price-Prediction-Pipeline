@echo off
echo Setting up environment variables for eBay API...
echo.
echo Please enter your eBay API credentials:
echo.

set /p EBAY_APP_ID="Enter your eBay App ID: "
set /p EBAY_CLIENT_SECRET="Enter your eBay Client Secret: "

echo.
echo Environment variables set successfully!
echo.
echo EBAY_APP_ID=%EBAY_APP_ID%
echo EBAY_CLIENT_SECRET=***hidden***
echo.

REM Export to system (optional - requires admin)
REM setx EBAY_APP_ID "%EBAY_APP_ID%"
REM setx EBAY_CLIENT_SECRET "%EBAY_CLIENT_SECRET%"

echo.
echo Note: These variables are set for this session only.
echo To make them permanent, run this script as administrator.
echo.
pause
