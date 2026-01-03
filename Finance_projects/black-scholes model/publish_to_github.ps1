# PowerShell script to publish Black-Scholes Model to GitHub
# Run this script from the project directory

Write-Host "Setting up Git repository for Black-Scholes Model..." -ForegroundColor Green

# Initialize git
Write-Host "`n1. Initializing git repository..." -ForegroundColor Yellow
git init

# Add all files
Write-Host "`n2. Adding all files..." -ForegroundColor Yellow
git add .

# Create initial commit
Write-Host "`n3. Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial commit: Black-Scholes Model - Comprehensive option pricing with Greeks, Monte Carlo, and implied volatility"

Write-Host "`n✅ Local repository ready!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Create a repository on GitHub named 'quant-finance' (or your preferred name)" -ForegroundColor White
Write-Host "2. Run these commands:" -ForegroundColor White
Write-Host "   git remote add origin https://github.com/YOUR_USERNAME/quant-finance.git" -ForegroundColor Gray
Write-Host "   git branch -M main" -ForegroundColor Gray
Write-Host "   git push -u origin main" -ForegroundColor Gray
Write-Host "`nNote: Replace YOUR_USERNAME with your GitHub username" -ForegroundColor Yellow

