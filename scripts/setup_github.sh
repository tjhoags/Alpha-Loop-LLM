#!/bin/bash
# ALC-Algo GitHub Setup Script (Linux/Mac)

echo "================================================================="
echo "ALC-ALGO GITHUB SETUP"
echo "================================================================="

# 1. Check Git
if ! command -v git &> /dev/null; then
    echo "Git not found! Please install Git."
    exit 1
fi

# 2. Check if already initialized
if [ -d ".git" ]; then
    echo "Repository already initialized."
    read -p "Do you want to re-initialize? (y/n) " choice
    if [ "$choice" != "y" ]; then
        exit 0
    fi
fi

# 3. Initialize
echo "Initializing Git repository..."
git init -b main

# 4. Check .gitignore
if [ ! -f ".gitignore" ]; then
    echo ".gitignore not found! Creating default..."
    echo -e "config/secrets.py\n.env\ndata/\n__pycache__/\n*.log" > .gitignore
fi

# 5. Add and Commit
echo "Adding files..."
git add .
git commit -m "feat: Initial setup of ALC-Algo Institutional Platform"

# 6. Prompt for Remote
read -p "Enter GitHub Repository URL (e.g., https://github.com/User/Repo.git): " remote
if [ -n "$remote" ]; then
    if git remote | grep -q "origin"; then
        git remote remove origin
    fi
    git remote add origin "$remote"
    echo "Remote 'origin' added."
    
    echo "Pushing to main..."
    git push -u origin main
    
    echo "Creating 'dev' branch..."
    git checkout -b dev
    git push -u origin dev
    
    echo "Returning to 'dev' branch."
else
    echo "Skipping remote setup. Run 'git remote add origin <url>' manually."
fi

echo "================================================================="
echo "SETUP COMPLETE"
echo "================================================================="

