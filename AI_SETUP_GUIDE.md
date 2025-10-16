# AI Integration Setup Guide

The Astro-AI platform supports multiple AI providers for enhanced analysis capabilities.

## 🤖 **Supported AI Providers**

### 1. OpenRouter (Recommended - Free Option Available)
- **Model**: DeepSeek R1 (`deepseek/deepseek-r1:free`)
- **Cost**: Free tier available
- **Performance**: Excellent for scientific analysis
- **Setup**: See instructions below

### 2. OpenAI
- **Models**: GPT-4o, GPT-3.5-turbo
- **Cost**: Pay-per-use
- **Performance**: Industry standard
- **Setup**: Requires OpenAI account

## 🔧 **OpenRouter Setup (Recommended)**

### Step 1: Get OpenRouter API Key
1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Sign up for a free account
3. Navigate to your dashboard
4. Generate an API key (starts with `sk-or-v1-...`)
5. Copy your API key

### Step 2: Configure in Streamlit Cloud
1. Go to your Streamlit Cloud dashboard
2. Select your app: `cam-sust2025`
3. Click **"Settings"** (gear icon)
4. Click **"Secrets"** tab
5. Add the following secrets:

```toml
OPENROUTER_API_KEY = "sk-or-v1-your-actual-key-here"
OPENROUTER_MODEL = "deepseek/deepseek-r1:free"
```

### Step 3: Deploy & Verify
1. Save the secrets
2. Your app will automatically restart
3. Look for success message: "✅ OpenRouter AI enabled with model: deepseek/deepseek-r1:free"

## 🔧 **OpenAI Setup (Alternative)**

### Step 1: Get OpenAI API Key
1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Sign up and add billing information
3. Navigate to API Keys section
4. Generate an API key (starts with `sk-...`)
5. Copy your API key

### Step 2: Configure in Streamlit Cloud
Add this secret:

```toml
OPENAI_API_KEY = "sk-your-actual-openai-key-here"
```

## 📍 **Where to Set Secrets in Streamlit Cloud**

### Method 1: Streamlit Cloud Dashboard
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Find your app: `cam-sust2025`
3. Click the **⚙️ Settings** button
4. Select **"Secrets"** from the sidebar
5. Add your secrets in TOML format

### Method 2: Direct App Settings
1. While viewing your deployed app
2. Look for **"Manage app"** link (usually in bottom right)
3. Click **"Settings"**
4. Navigate to **"Secrets"** tab
5. Paste your configuration

## 🎯 **Example Secret Configuration**

```toml
# For OpenRouter (Free DeepSeek R1)
OPENROUTER_API_KEY = "sk-or-v1-1234567890abcdef..."
OPENROUTER_MODEL = "deepseek/deepseek-r1:free"

# Alternative: For OpenAI (Paid)
# OPENAI_API_KEY = "sk-1234567890abcdef..."
```

## ✅ **Verification**

After setting up your secrets:

1. **Restart Required**: Streamlit will automatically restart your app
2. **Success Indicators**: 
   - Look for green success message in the app
   - AI features will switch from "Simulation Mode" to real AI
3. **Features Enabled**:
   - ✨ AI-powered scientific insights
   - 📊 Intelligent comparative analysis  
   - 📝 Automated report generation
   - 🎯 Smart next-step suggestions

## 🆓 **Why OpenRouter + DeepSeek R1?**

- **Free Tier**: No costs for moderate usage
- **High Quality**: DeepSeek R1 is optimized for reasoning and analysis
- **Fast**: Quick response times
- **Scientific Focus**: Excellent performance on technical content
- **No Billing Required**: Get started immediately

## 🔄 **Switching Providers**

You can switch between providers by updating your secrets:

```toml
# Switch to OpenAI
OPENAI_API_KEY = "your-openai-key"
# Remove or comment out OpenRouter keys

# Switch back to OpenRouter  
OPENROUTER_API_KEY = "your-openrouter-key"
OPENROUTER_MODEL = "deepseek/deepseek-r1:free"
# Remove or comment out OpenAI key
```

## 🛠️ **Troubleshooting**

### Issue: Still seeing "Simulation Mode"
**Solution**: 
1. Verify secrets are correctly formatted (TOML syntax)
2. Check that keys don't have extra spaces
3. Restart the app manually if needed

### Issue: API errors
**Solution**:
1. Verify your API key is valid and active
2. Check your account has sufficient credits (OpenAI) or quota (OpenRouter)
3. Try a different model if available

### Issue: Secrets not updating
**Solution**:
1. Clear browser cache
2. Wait a few minutes for Streamlit Cloud to propagate changes
3. Check the app logs for error messages

## 📞 **Support**

If you encounter issues:
1. Check the Streamlit app logs for error messages
2. Verify your API key is correctly formatted
3. Try the simulation mode first to ensure other features work
4. Contact support if problems persist

---

**Ready to enable AI-powered astronomical analysis!** 🚀