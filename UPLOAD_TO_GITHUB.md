# 📤 How to Upload to GitHub

## Step 1: Create Repository on GitHub (2 minutes)

1. Go to https://github.com/Keval-7503
2. Click the **green "New"** button (or go to https://github.com/new)
3. Repository name: `compuer_vision_assignment4`
4. Description: `Assignment 4: Augmented Reality with PyTorch3D`
5. Make it **Public** ✅
6. **DO NOT** check "Add README" (we already have one)
7. Click **"Create repository"**

---

## Step 2: Upload Using Git (5 minutes)

### Open Terminal in VS Code

Press `` Ctrl+` `` (backtick key) or View → Terminal

### Run These Commands (Copy-Paste):

```bash
# Navigate to your project
cd E:\compuer_vision_assignment4

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Assignment 4: Augmented Reality with PyTorch3D"

# Connect to GitHub
git remote add origin https://github.com/Keval-7503/compuer_vision_assignment4.git

# Push to GitHub
git branch -M main
git push -u origin main
```

If it asks for credentials:
- Username: `Keval-7503`
- Password: Use your **Personal Access Token** (not your GitHub password)

### Don't have a Personal Access Token?

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Give it a name: "Assignment4"
4. Check: `repo` (Full control of private repositories)
5. Click "Generate token"
6. **COPY THE TOKEN** (you won't see it again!)
7. Use this token as your password when pushing

---

## Step 3: Verify Upload (1 minute)

1. Go to: https://github.com/Keval-7503/compuer_vision_assignment4
2. You should see all your files!
3. Click on `Assignment4_AR_PyTorch3D.ipynb` - it should display

---

## Step 4: Test in Google Colab (5 minutes)

1. Go to: https://colab.research.google.com/
2. File → Open Notebook → GitHub tab
3. Enter: `Keval-7503/compuer_vision_assignment4`
4. Click on `Assignment4_AR_PyTorch3D.ipynb`
5. Run the first cell (setup)
6. Click "Runtime" → "Run all"
7. Wait for it to complete ✅

---

## Alternative: Upload Using GitHub Desktop (Easier!)

### Download GitHub Desktop

https://desktop.github.com/

### Steps:

1. Open GitHub Desktop
2. File → Add Local Repository
3. Choose: `E:\compuer_vision_assignment4`
4. Click "Create Repository" or "Add Repository"
5. Click "Publish repository" button
6. Uncheck "Keep this code private"
7. Click "Publish repository"

Done! ✅

---

## Alternative: Upload Using GitHub Website (No Git Required!)

### If Git is too complicated:

1. Go to: https://github.com/new
2. Create repository: `compuer_vision_assignment4` (Public)
3. Click "Create repository"
4. Click "uploading an existing file"
5. Drag and drop ALL files from `E:\compuer_vision_assignment4`
6. Click "Commit changes"

⚠️ **Important**: Upload the `src/` folder by:
   - Click "Add file" → "Upload files"
   - Drag the entire `src` folder
   - Commit

---

## ✅ Success Check

Visit: https://github.com/Keval-7503/compuer_vision_assignment4

You should see:
- ✅ README.md displaying nicely
- ✅ Assignment4_AR_PyTorch3D.ipynb
- ✅ src/ folder with all .py files
- ✅ PROJECT_OVERVIEW.md
- ✅ requirements.txt

---

## 🎓 Submit on Canvas

Submit this URL:
```
https://github.com/Keval-7503/compuer_vision_assignment4
```

---

## 🆘 Having Issues?

### Issue: "repository already exists"
Delete it on GitHub first:
1. Go to: https://github.com/Keval-7503/compuer_vision_assignment4/settings
2. Scroll down → "Delete this repository"
3. Try again

### Issue: "failed to push"
```bash
git pull origin main --allow-unrelated-histories
git push origin main
```

### Issue: "Authentication failed"
Use Personal Access Token instead of password (see above)

---

## 🎉 You're Done!

Once uploaded, your assignment is ready to submit! 🚀
