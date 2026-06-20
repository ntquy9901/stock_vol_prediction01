#!/bin/bash
# GitHub Repository Setup & Push Script
# Usage: bash setup_github_repo.sh YOUR_USERNAME

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if username provided
if [ -z "$1" ]; then
    print_error "GitHub username not provided!"
    echo ""
    echo "Usage: bash setup_github_repo.sh YOUR_USERNAME"
    echo ""
    echo "Example:"
    echo "  bash setup_github_repo.sh johndoe"
    echo ""
    exit 1
fi

GITHUB_USERNAME=$1
REPO_NAME="stock_vol_prediction01"
REPO_URL="https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

print_step "GitHub Repository Setup Script"
echo ""
echo "GitHub Username: ${GITHUB_USERNAME}"
echo "Repository Name: ${REPO_NAME}"
echo "Repository URL: ${REPO_URL}"
echo ""

# Confirm
read -p "Continue with these settings? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_error "Aborted by user"
    exit 1
fi

# Step 1: Check if remote exists
print_step "Checking existing git remotes..."
if git remote get-url origin &>/dev/null; then
    print_warning "Remote 'origin' already exists:"
    git remote -v
    echo ""
    read -p "Remove and re-add? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote remove origin
        print_success "Removed old remote"
    else
        print_error "Aborted"
        exit 1
    fi
fi

# Step 2: Add remote
print_step "Adding GitHub remote..."
git remote add origin "${REPO_URL}"
print_success "Remote added: ${REPO_URL}"

# Step 3: Verify commits
print_step "Checking commits to push..."
COMMIT_COUNT=$(git log --oneline | wc -l)
echo "Found ${COMMIT_COUNT} commits:"
git log --oneline -3
echo ""

# Step 4: Show repository stats
print_step "Repository statistics..."
FILES_COUNT=$(git ls-files | wc -l)
CODE_LINES=$(git ls-files | xargs wc -l | tail -1 | awk '{print $1}')
echo "Total files: ${FILES_COUNT}"
echo "Total lines of code: ${CODE_LINES}"
echo ""

# Step 5: Push
print_step "Pushing to GitHub..."
echo "This will require authentication."
echo ""
echo "IMPORTANT: GitHub no longer accepts passwords!"
echo "You need a Personal Access Token:"
echo "  1. Go to: https://github.com/settings/tokens"
echo "  2. Click 'Generate new token (classic)'"
echo "  3. Check 'repo' scope"
echo "  4. Generate and copy the token"
echo ""
print_warning "When prompted for password:"
print_warning "  Username: ${GITHUB_USERNAME}"
print_warning "  Password: <PASTE YOUR TOKEN HERE>"
echo ""
read -p "Ready to push? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_error "Aborted by user"
    exit 1
fi

# Push with upstream
if git push -u origin master; then
    echo ""
    print_success "Successfully pushed to GitHub!"
    echo ""
    echo "Repository URL: https://github.com/${GITHUB_USERNAME}/${REPO_NAME}"
    echo ""
    echo "Next steps:"
    echo "  1. Visit your repository on GitHub"
    echo "  2. Verify all files are uploaded"
    echo "  3. Clone on Google Colab:"
    echo "     !git clone ${REPO_URL}"
    echo ""
else
    print_error "Push failed!"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Make sure you created the repository on GitHub"
    echo "  2. Verify your Personal Access Token has 'repo' scope"
    echo "  3. Check that the repository name is correct"
    echo ""
    echo "Repository creation link:"
    echo "  https://github.com/new"
    echo ""
    exit 1
fi
