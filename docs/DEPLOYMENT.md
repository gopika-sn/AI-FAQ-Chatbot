# Streamlit Community Cloud Deployment Guide

This document explains how to deploy **ShopAssist AI** online using **Streamlit Community Cloud** (100% free hosting).

---

## 🛠️ Step-by-Step Deployment Instructions

### Step 1: Create a GitHub Repository
1. Log into your [GitHub](https://github.com/) account.
2. Click **New Repository**.
3. Name the repository `AI-FAQ-Chatbot`.
4. Set repository visibility to **Public** or **Private**.
5. Do NOT initialize with a README (as we already have a complete `README.md`).

### Step 2: Push Project Code to GitHub
Open your terminal in the project directory and run:
```bash
git init
git add .
git commit -m "Initial commit of ShopAssist AI chatbot"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/AI-FAQ-Chatbot.git
git push -u origin main
```

> ⚠️ **SECURITY VERIFICATION**: Ensure `.env` is NOT uploaded to GitHub. Check `.gitignore` to confirm `.env` is excluded.

---

### Step 3: Deploy on Streamlit Community Cloud
1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
2. Click **New App** -> **Use existing repo**.
3. Select your repository: `YOUR_USERNAME/AI-FAQ-Chatbot`.
4. Set **Main file path** to `app.py`.
5. Set **Branch** to `main`.

---

### Step 4: Configure Gemini API Key Secrets
1. Before clicking Deploy, expand **Advanced Settings** -> **Secrets**.
2. Add your Gemini API key under Streamlit Secrets format:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key_here"
   ```
3. Click **Save**.

---

### Step 5: Launch & Test Application
1. Click **Deploy!**
2. Streamlit will automatically build the environment, install `requirements.txt`, download the spaCy `en_core_web_sm` model, and launch the web app.
3. Test both **📚 FAQ-based responses** and **✨ AI-generated responses** on your live URL!
