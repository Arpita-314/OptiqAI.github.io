# Quick Commands to Push to GitHub

## After Creating Repository on GitHub

Run these commands in order:

```bash
cd D:\code\foaml\Finance_projects\black_scholes

# 1. Add your GitHub repository as remote (replace YOUR_USERNAME and REPO_NAME)
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 2. Rename branch to main (if needed)
git branch -M main

# 3. Push to GitHub
git push -u origin main
```

## If Repository Already Exists

If you already have a remote, check and update:

```bash
# Check current remotes
git remote -v

# Update remote URL if needed
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# Push
git push -u origin main
```

## Authentication

If prompted for credentials:
- **Personal Access Token**: Use a GitHub Personal Access Token (not password)
- **SSH**: If using SSH, ensure your SSH key is set up with GitHub

## Verify Upload

After pushing, visit your GitHub repository to verify all files are uploaded correctly.

