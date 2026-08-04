# Deploying CoverSignal on Oracle Cloud (Free Tier)

This guide walks through a **full production deployment** of CoverSignal on an
**Oracle Cloud Always Free** ARM VM with a **public HTTPS URL**, sized for low
traffic (~10 users/month).

**Architecture:**

```text
Browser  →  Cloudflare (free HTTPS + DNS)  →  Oracle VM
                                              ├ nginx :80
                                              │   ├ /        → frontend/dist (React SPA)
                                              │   └ /api/…   → uvicorn :8000 (FastAPI)
                                              ├ SQLite + uploads in data/
                                              └ ML models in backend/models/ + models/
```

**Estimated cost:** **$0/month** on Oracle Always Free + Cloudflare Free, plus
**~$1–12/year** if you buy a domain (optional but recommended for a clean URL).

---

## What you need before starting

| Item | Notes |
|------|-------|
| Oracle Cloud account | Credit card required for verification; Always Free resources stay $0 |
| Domain name (recommended) | e.g. `coversignal.yourdomain.com` via Cloudflare (~$10/yr for `.com`, or use a subdomain you already own) |
| SSH client | Windows: PowerShell / Git Bash / PuTTY |
| Local machine | Your dev copy with trained model artifacts (`backend/models/*.pkl`) |

> **Note:** Model `.pkl` files are gitignored. You must copy them from your
> local machine or re-run training on the server (requires `engagement.csv` +
> `downloads/`, which are also gitignored).

---

## Part 1 — Create the Oracle Cloud VM

### 1.1 Sign up and create a compartment

1. Go to [https://cloud.oracle.com](https://cloud.oracle.com) and create an account.
2. After login, open the **hamburger menu → Identity & Security → Compartments**.
3. Create a compartment (e.g. `coversignal`) or use the root compartment.

### 1.2 Create an Always Free ARM instance

1. **Menu → Compute → Instances → Create instance**.
2. **Name:** `coversignal`
3. **Compartment:** your compartment
4. **Placement:** keep default AD in your home region (pick a region close to you).
5. **Image:** **Ubuntu 22.04** (aarch64 / ARM)
6. **Shape:** Click **Change shape**
   - Select **Ampere** (ARM)
   - Pick **VM.Standard.A1.Flex**
   - **OCPUs:** 2 (uses less of your free quota; enough for this app)
   - **Memory:** 12 GB (or 24 GB if you have quota — more headroom for video analysis)
7. **Networking:** Create new VCN if prompted (defaults are fine).
8. **Public IP:** Assign a **public IPv4 address**.
9. **SSH keys:** Generate or upload your public key (recommended).
   - Windows (PowerShell): `ssh-keygen -t ed25519` → upload `~/.ssh/id_ed25519.pub`
10. **Boot volume:** 50 GB is plenty.
11. Click **Create**.

Wait until the instance state is **Running**. Copy the **Public IP address**.

### 1.3 Open firewall ports (Oracle + OS)

**A. Oracle Security List (cloud firewall)**

1. **Menu → Networking → Virtual cloud networks** → your VCN.
2. Click the **Security List** for your public subnet.
3. **Add Ingress Rules:**

| Source CIDR | Protocol | Dest Port | Description |
|-------------|----------|-----------|-------------|
| `0.0.0.0/0` | TCP | 22 | SSH |
| `0.0.0.0/0` | TCP | 80 | HTTP |
| `0.0.0.0/0` | TCP | 443 | HTTPS (optional if using Cloudflare Flexible SSL on :80) |

**B. Ubuntu firewall (after SSH in)**

```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save   # if available
```

Oracle Ubuntu images often have `iptables` rules that block 80/443 by default.

---

## Part 2 — Public URL with Cloudflare (free HTTPS)

You need a domain. Easiest path:

1. Buy a domain at [Cloudflare Registrar](https://www.cloudflare.com/products/registrar/)
   (at-cost pricing, often ~$10/year for `.com`).
2. Or use a subdomain of a domain you already manage in Cloudflare
   (e.g. `coversignal.yourname.dev`).

### 2.1 Point DNS to your VM

1. In Cloudflare dashboard → your domain → **DNS → Records**.
2. Add an **A record:**
   - **Name:** `coversignal` (or `@` for apex)
   - **IPv4:** your Oracle VM public IP
   - **Proxy status:** **Proxied** (orange cloud) — enables free HTTPS
3. Save. DNS may take a few minutes to propagate.

Your public URL will be: `https://coversignal.yourdomain.com`

### 2.2 SSL mode

For the simplest setup (no cert on the VM):

1. **SSL/TLS → Overview → Configure**
2. Set encryption mode to **Flexible**
   - Visitor → Cloudflare: HTTPS
   - Cloudflare → your VM: HTTP on port 80

This avoids installing Let's Encrypt on the VM. For production-hardening later,
switch to **Full (strict)** and add a cert with Caddy or Certbot.

---

## Part 3 — Server setup (SSH into the VM)

```bash
ssh ubuntu@YOUR_VM_PUBLIC_IP
```

All commands below run on the VM unless noted.

### 3.1 System packages

```bash
sudo apt update && sudo apt upgrade -y

sudo apt install -y \
  python3.11 python3.11-venv python3-pip \
  nginx git \
  libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
  netfilter-persistent
```

OpenCV/MediaPipe need those graphics libs even on a headless server.

### 3.2 Install Node.js (to build the frontend on-server)

```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version   # should print v20.x
```

Alternatively, build `frontend/dist` on your laptop and `scp` it to the VM.

### 3.3 Clone the repo

```bash
cd ~
git clone https://github.com/GeneWC/tiktok-cover-analysis.git tiktokcover
cd tiktokcover
```

---

## Part 4 — Python backend

### 4.1 Create virtualenv and install dependencies

```bash
cd ~/tiktokcover
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> First `pip install` on ARM may take 5–10 minutes (OpenCV, MediaPipe, etc.).

### 4.2 Copy model artifacts from your local machine

On **your laptop** (not the VM), from the project root:

```bash
scp backend/models/*.pkl backend/models/*.json ubuntu@YOUR_VM_IP:~/tiktokcover/backend/models/
```

Verify on the VM:

```bash
ls ~/tiktokcover/backend/models/
# Expect: *.pkl, calibration.json, feature_schema.json, etc.
```

If `.pkl` files are missing, the API will start but analysis predictions will fail.

### 4.3 Production environment file

Create `~/tiktokcover/.env`:

```bash
cat > ~/tiktokcover/.env << 'EOF'
# Allow your public frontend origin (same-origin deploy still helps for dev/testing)
COVERSIGNAL_CORS_ALLOW_ORIGINS=["https://coversignal.yourdomain.com"]
EOF
```

Replace `coversignal.yourdomain.com` with your actual domain.

### 4.4 Smoke-test the backend

```bash
cd ~/tiktokcover
source .venv/bin/activate
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

In another SSH session (or from laptop with SSH tunnel):

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/analyze/ping
```

Press `Ctrl+C` to stop. MediaPipe/OCR model files download automatically on first
real analysis (needs outbound internet).

---

## Part 5 — Frontend (production build)

Build with your public API URL (same domain as the site — nginx proxies `/api`):

```bash
cd ~/tiktokcover/frontend

cat > .env.production << 'EOF'
VITE_API_BASE_URL=https://coversignal.yourdomain.com
EOF

npm install
npm run build
```

Output goes to `frontend/dist/`. Confirm:

```bash
ls dist/index.html
```

---

## Part 6 — systemd service (keep backend running)

```bash
sudo tee /etc/systemd/system/coversignal.service << 'EOF'
[Unit]
Description=CoverSignal FastAPI backend
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/tiktokcover
EnvironmentFile=/home/ubuntu/tiktokcover/.env
ExecStart=/home/ubuntu/tiktokcover/.venv/bin/uvicorn backend.app:app --host 127.0.0.1 --port 8000 --workers 1
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable coversignal
sudo systemctl start coversignal
sudo systemctl status coversignal
```

Use **1 worker** — video analysis is CPU-heavy; multiple workers on a 2-OCPU VM
can cause OOM/timeouts. For ~10 users/month this is fine.

Logs:

```bash
journalctl -u coversignal -f
```

---

## Part 7 — nginx (serve frontend + proxy API)

```bash
sudo tee /etc/nginx/sites-available/coversignal << 'EOF'
server {
    listen 80;
    server_name coversignal.yourdomain.com;

    client_max_body_size 200M;

    root /home/ubuntu/tiktokcover/frontend/dist;
    index index.html;

    # FastAPI backend
    location /api/ {
        proxy_pass         http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }

    # OpenAPI docs (optional — remove if you don't want public /docs)
    location /docs {
        proxy_pass http://127.0.0.1:8000/docs;
    }
    location /openapi.json {
        proxy_pass http://127.0.0.1:8000/openapi.json;
    }
    location /health {
        proxy_pass http://127.0.0.1:8000/health;
    }

    # React SPA — client-side routes
    location / {
        try_files $uri $uri/ /index.html;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/coversignal /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl reload nginx
```

Replace `coversignal.yourdomain.com` in the `server_name` line.

---

## Part 8 — Verify end-to-end

1. Open `https://coversignal.yourdomain.com` in a browser.
2. Upload a short test video (< 120 s, `.mp4`).
3. Watch the processing page poll until complete.
4. Confirm the report renders with scores and tiers.

**Quick checks if something fails:**

| Symptom | Likely fix |
|---------|------------|
| 502 Bad Gateway | `sudo systemctl status coversignal` — backend not running |
| CORS error in browser | Update `COVERSIGNAL_CORS_ALLOW_ORIGINS` in `.env`, restart service |
| 413 Request Entity Too Large | Confirm `client_max_body_size 200M` in nginx |
| Analysis stuck / failed | `journalctl -u coversignal -f` — missing `.pkl` or MediaPipe model download failed |
| Site loads but API 404 | `VITE_API_BASE_URL` must match your domain; rebuild frontend |
| Connection refused on :80 | Oracle security list + `iptables` rules from §1.3 |

---

## Part 9 — Deploy updates (after you push to GitHub)

On the VM:

```bash
cd ~/tiktokcover
git pull

source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
npm run build

sudo systemctl restart coversignal
sudo systemctl reload nginx
```

If you changed model artifacts locally, re-`scp` the `backend/models/` files.

---

## Part 10 — Maintenance for a 6-month run

### Disk space

Uploads and reports accumulate under `data/`:

```bash
du -sh ~/tiktokcover/data/*
```

Periodically prune old uploads/reports if needed:

```bash
# Example: delete reports older than 30 days
find ~/tiktokcover/data/reports -type f -mtime +30 -delete
find ~/tiktokcover/data/videos -type f -mtime +30 -delete
```

### Auto security updates (optional)

```bash
sudo apt install unattended-upgrades
sudo dpkg-reconfigure -plow unattended-upgrades
```

### Backups (optional)

For 10 users/month, a weekly tarball is enough:

```bash
tar czf ~/coversignal-backup-$(date +%F).tar.gz \
  ~/tiktokcover/data/coversignal.db \
  ~/tiktokcover/backend/models/
```

### Monitoring

Oracle free tier has basic metrics in the console. For uptime checks, use
[UptimeRobot](https://uptimerobot.com) (free) to ping `https://yourdomain.com/health`.

---

## Cost summary

| Service | Cost |
|---------|------|
| Oracle VM.Standard.A1.Flex (Always Free) | $0/mo |
| Cloudflare DNS + HTTPS proxy | $0/mo |
| Domain (optional) | ~$1–12/year |
| **Total for 6 months** | **$0–6** |

---

## Troubleshooting Oracle-specific issues

**"Out of host capacity" when creating ARM instance**
Try a different availability domain or region, or retry later. ARM free capacity
is limited in popular regions.

**Instance stops / idle reclaim**
Oracle may reclaim idle Always Free resources on rarely-used accounts. Log in
monthly and keep the service running to reduce risk.

**Can't SSH**
Confirm your public key was added at instance creation and port 22 is open in
the security list.

**Slow first analysis**
The first video triggers MediaPipe + EAST model downloads into `models/`. This is
normal and only happens once.

---

## Security notes for public use

- Consider removing public `/docs` in nginx when you don't need them.
- Rate limiting is not built into the MVP — at 10 users/month you're fine, but
  add nginx `limit_req` or Cloudflare rate rules if you ever open it wider.
- Do not commit `.env`, datasets, or model training data (already in `.gitignore`).

---

## Quick reference

| What | Where |
|------|-------|
| App URL | `https://coversignal.yourdomain.com` |
| API | `https://coversignal.yourdomain.com/api/analyze` |
| Health | `https://coversignal.yourdomain.com/health` |
| Backend logs | `journalctl -u coversignal -f` |
| nginx config | `/etc/nginx/sites-available/coversignal` |
| Env file | `~/tiktokcover/.env` |
| Uploads / DB | `~/tiktokcover/data/` |
