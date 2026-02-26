# 🚀 SENTINEL — DigitalOcean Deployment Guide

## Architecture

```
┌─────────────────────────────────────────┐
│  DigitalOcean Droplet (Singapore)       │
│  Ubuntu 24.04 · $6/mo · 1GB RAM        │
│                                         │
│  ┌──────────────┐  ┌────────────────┐   │
│  │  Python Bot   │  │  Node.js       │   │
│  │  (main.py)    │  │  Dashboard     │   │
│  │              │◄─►│  :3001         │   │
│  └──────┬───────┘  └───────┬────────┘   │
│         │    Shared Volume  │            │
│         └────► /app/data ◄──┘            │
│                                         │
│  Port 3001 ──► Internet (your browser)  │
└─────────────────────────────────────────┘
```

> **💡 Use Singapore (sgp1) datacenter** — lowest latency to Binance servers.

---

## Step 1: Create Droplet

1. Go to [cloud.digitalocean.com](https://cloud.digitalocean.com)
2. Create Droplet:
   - **Image:** Ubuntu 24.04 LTS
   - **Plan:** Basic → $6/mo (1 vCPU, 1GB RAM, 25GB SSD)
   - **Region:** Singapore (sgp1)
   - **Auth:** SSH key (recommended) or password
3. Note your droplet's **IP address**

---

## Step 2: SSH Into Droplet & Install Docker

```bash
# SSH into your droplet
ssh root@YOUR_DROPLET_IP

# Install Docker
curl -fsSL https://get.docker.com | sh

# Install Docker Compose
apt install -y docker-compose-plugin

# Verify
docker --version
docker compose version
```

---

## Step 3: Upload Your Code

**Option A — Git (recommended):**
```bash
# On your Mac, push to a private repo first
cd "/Users/nikhildhawan/Documents/synaptic/Algocrypto/Cloned algo/HMMBOT"
git init
git add .
git commit -m "Initial commit"
git remote add origin git@github.com:YOUR_USERNAME/HMMBOT.git
git push -u origin main

# On the droplet
git clone git@github.com:YOUR_USERNAME/HMMBOT.git
cd HMMBOT
```

**Option B — SCP (direct upload):**
```bash
# From your Mac terminal
scp -r "/Users/nikhildhawan/Documents/synaptic/Algocrypto/Cloned algo/HMMBOT" \
    root@YOUR_DROPLET_IP:/root/HMMBOT
```

---

## Step 4: Configure Environment

```bash
cd /root/HMMBOT

# Create .env with your real credentials
cat > .env << 'EOF'
BINANCE_API_KEY=your_real_api_key
BINANCE_API_SECRET=your_real_api_secret
TESTNET=false
PAPER_TRADE=true
EOF

# Create data directory
mkdir -p data
```

---

## Step 5: Build & Run

```bash
# Build both containers
docker compose build

# Start in background (detached)
docker compose up -d

# Check status
docker compose ps

# View bot logs
docker compose logs -f bot

# View dashboard logs
docker compose logs -f dashboard
```

---

## Step 6: Access Dashboard

Open in browser:
```
http://YOUR_DROPLET_IP:3001
```

---

## Useful Commands

```bash
# Restart everything
docker compose restart

# Restart just the bot (picks up config changes)
docker compose restart bot

# Stop everything
docker compose down

# Rebuild after code changes
docker compose build && docker compose up -d

# View real-time bot logs
docker compose logs -f bot --tail=50

# SSH into bot container for debugging
docker compose exec bot bash
```

---

## Optional: HTTPS with Nginx + Let's Encrypt

If you have a domain name and want HTTPS:

```bash
apt install -y nginx certbot python3-certbot-nginx

# Create nginx config
cat > /etc/nginx/sites-available/sentinel << 'EOF'
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:3001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

ln -s /etc/nginx/sites-available/sentinel /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# Get SSL certificate
certbot --nginx -d your-domain.com
```

---

## Monthly Cost

| Component | Cost |
|-----------|------|
| Droplet (1GB, SGP) | $6/mo |
| Domain (optional) | ~$10/yr |
| **Total** | **~$6/mo** |
