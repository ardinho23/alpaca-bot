import os
API_KEY    = os.environ.get("PKGCSLJCUZQPVUDOEDYPJRA3GC")
SECRET_KEY = os.environ.get("D8YWgfM1DSpC5rgecx82ABbFP8BpJjXLsGpZamv7Q5KF")
```

---

## Step 4 — Deploy on Railway
- Go to **[railway.app](https://railway.app)**
- Sign up with your GitHub account
- Click **"New Project"** → **"Deploy from GitHub repo"**
- Select your `alpaca-bot` repo
- Railway auto-detects and starts deploying

---

## Step 5 — Add API keys
In Railway dashboard → your project → **Variables** tab → add:
```
APCA_API_KEY_ID      → your key id
APCA_API_SECRET_KEY  → your secret key
PAPER                → true
