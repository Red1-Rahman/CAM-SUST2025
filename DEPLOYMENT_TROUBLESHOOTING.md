# 🚨 Streamlit Cloud Deployment Troubleshooting

## Common "Error installing requirements" Solutions

### 🎯 **Quick Fix - Use Minimal Requirements**

If you're getting installation errors, try this step-by-step approach:

#### Step 1: Deploy with Minimal Requirements

Use the current `requirements.txt` which has been optimized for Streamlit Cloud with:

- ✅ Fixed versions (not >=)
- ✅ Core packages only
- ✅ Heavy packages commented out

#### Step 2: Verify Basic Deployment

Check that the app loads with:

- Streamlit framework
- Basic scientific computing (numpy, pandas, scipy)
- Matplotlib visualization
- Astropy astronomy core

#### Step 3: Gradually Add Features

If basic deployment works, uncomment packages one by one:

```bash
# Try adding these one at a time:
bagpipes==1.0.0
corner==2.2.1
```

### 🔧 **Common Package Issues**

| Package         | Issue                    | Solution                                         |
| --------------- | ------------------------ | ------------------------------------------------ |
| `bagpipes`      | Compilation errors       | Use specific version `bagpipes==1.0.0` or remove |
| `corner`        | C++ compiler needed      | Remove for now, app has fallback                 |
| `py21cmfast`    | System libraries missing | Keep commented - requires packages.txt           |
| `jwst`          | Very heavy (>1GB)        | Keep commented - timeout issues                  |
| `openai>=1.0.0` | Version conflicts        | Use exact version `openai==1.3.5`                |

### 🐛 **Specific Error Debugging**

#### "Could not install packages due to an EnvironmentError"

**Solution**: Package version conflicts

```bash
# Replace >= with == for specific versions
streamlit==1.28.1
numpy==1.24.3
pandas==2.0.3
```

#### "Building wheel for [package] failed"

**Solution**: Remove packages that need compilation

```bash
# Comment out these in requirements.txt:
# bagpipes>=1.0.0
# corner>=2.2.0
```

#### "No space left on device"

**Solution**: Too many/too large packages

```bash
# Remove development packages:
# pytest>=7.0.0
# black>=23.0.0

# Remove heavy scientific packages:
# jwst>=1.12.0
```

#### "Timeout during pip install"

**Solution**: Streamlit Cloud has build time limits

```bash
# Keep only essential packages:
streamlit
numpy
pandas
matplotlib
astropy
openai
```

### 🚀 **Recommended Deployment Sequence**

1. **First Deploy**: Use current `requirements.txt` (minimal)
2. **Verify**: App loads and shows status panel
3. **Test Core**: Data upload, basic analysis work
4. **Add Features**: Uncomment packages gradually
5. **Monitor**: Check build logs for any issues

### 📋 **Deployment Checklist**

- ✅ Python version: 3.10
- ✅ Main file: `app.py`
- ✅ Branch: `main`
- ✅ Requirements: Use minimal version first
- ✅ System packages: `packages.txt` present
- ✅ Setup script: `setup.sh` present

### 🔍 **Debug Your Current Error**

To help debug your specific error:

1. **Check Build Logs**: In Streamlit Cloud, click "View logs" during deployment
2. **Find Error Line**: Look for lines starting with "ERROR:" or "Failed"
3. **Identify Package**: Note which package is causing the failure
4. **Apply Fix**: Comment out the problematic package

### 💡 **Pro Tips**

- **Start Simple**: Deploy minimal version first, add complexity later
- **Monitor Logs**: Always check build logs for specific error messages
- **Version Lock**: Use exact versions (==) instead of minimum versions (>=)
- **Fallback Ready**: Your app is designed to work without heavy packages

### 🆘 **Emergency Minimal Requirements**

If nothing works, try this ultra-minimal `requirements.txt`:

```txt
streamlit==1.28.1
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2
```

This should always deploy successfully and give you a working base to build upon.

---

**Need Help?** Check the specific error message in Streamlit Cloud build logs and match it to the solutions above!
