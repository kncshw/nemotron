# HTTPS Reverse Proxy for vLLM + Open WebUI

> **Date:** 2026-04-14
> **Context:** GB10 running vLLM (port 8000) + Open WebUI (port 3000) in Docker
> **Goal:** Single HTTPS endpoint (port 443) serving both API and admin chat UI, with the chat UI restricted to admin access only

---

## Architecture

```
External clients (API consumers)
        │
        ▼
┌──────────────────────────────────┐
│  Reverse Proxy (nginx or Caddy)  │
│  :443 (HTTPS, self-signed TLS)   │
│                                   │
│  /v1/*  ──► localhost:8000 (vLLM) │   ← open to all
│  /*     ──► localhost:3000 (OWUI) │   ← admin IP only
└──────────────────────────────────┘
        │                    │
        ▼                    ▼
┌──────────────┐   ┌────────────────┐
│ vllm_node    │   │ open-webui     │
│ :8000 (HTTP) │   │ :3000 (HTTP)   │
└──────────────┘   └────────────────┘
```

| URL | Routes to | Access |
|---|---|---|
| `https://<gb10-ip>/v1/chat/completions` | vLLM API | Open to all |
| `https://<gb10-ip>/v1/models` | vLLM API | Open to all |
| `https://<gb10-ip>/` | Open WebUI | Admin IP only |

---

## Prerequisites

Both containers must be reachable from the host:

```bash
# Verify vLLM is serving
curl -s http://localhost:8000/v1/models

# Verify Open WebUI is serving
curl -s http://localhost:3000
```

---

## Option A: nginx

### 1. Generate self-signed certificate

```bash
mkdir -p ~/nginx/certs

openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout ~/nginx/certs/selfsigned.key \
  -out ~/nginx/certs/selfsigned.crt \
  -subj "/CN=<gb10-ip>"
```

Replace `<gb10-ip>` with the actual host IP (e.g. `172.16.100.x`).

### 2. Create nginx config

```bash
cat > ~/nginx/nginx.conf << 'CONF'
events {}

http {
    server {
        listen 443 ssl;
        ssl_certificate     /etc/nginx/certs/selfsigned.crt;
        ssl_certificate_key /etc/nginx/certs/selfsigned.key;

        # vLLM API — open to all
        location /v1/ {
            proxy_pass http://localhost:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_read_timeout 300s;
        }

        # Open WebUI — admin only
        location / {
            allow 127.0.0.1;
            allow <your-workstation-ip>;
            deny all;

            proxy_pass http://localhost:3000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
CONF
```

Replace `<your-workstation-ip>` with the IP you SSH from (not the GB10's IP).

**Key settings explained:**

| Setting | Why |
|---|---|
| `proxy_read_timeout 300s` | LLM responses can take minutes for long generations |
| `Upgrade` / `Connection "upgrade"` | Open WebUI uses WebSocket for streaming chat |
| `allow` / `deny all` | Restricts chat UI to admin IPs; everyone else gets 403 |

### 3. Run nginx

```bash
docker run -d --name nginx \
  --network host \
  --restart unless-stopped \
  -v ~/nginx/nginx.conf:/etc/nginx/nginx.conf:ro \
  -v ~/nginx/certs:/etc/nginx/certs:ro \
  nginx:latest
```

### 4. Verify

```bash
# API (should work from anywhere)
curl -k https://localhost/v1/models

# Chat UI (should work from allowed IPs, 403 from others)
curl -k https://localhost/
```

### 5. Managing nginx

```bash
# Reload config without restarting (after editing nginx.conf)
docker exec nginx nginx -s reload

# View logs
docker logs nginx

# Stop
docker stop nginx && docker rm nginx
```

---

## Option B: Caddy

Caddy auto-generates self-signed certs with `tls internal` — no openssl commands needed.

### 1. Create Caddyfile

```bash
mkdir -p ~/caddy

cat > ~/caddy/Caddyfile << 'CONF'
https://:443 {
    tls internal

    # vLLM API — open to all
    handle /v1/* {
        reverse_proxy localhost:8000
    }

    # Open WebUI — admin only
    handle {
        @blocked not remote_ip 127.0.0.1 <your-workstation-ip>
        respond @blocked 403

        reverse_proxy localhost:3000
    }
}
CONF
```

Replace `<your-workstation-ip>` with your workstation IP.

### 2. Run Caddy

```bash
docker run -d --name caddy \
  --network host \
  --restart unless-stopped \
  -v ~/caddy/Caddyfile:/etc/caddy/Caddyfile:ro \
  -v caddy_data:/data \
  caddy:latest
```

### 3. Verify

```bash
# API
curl -k https://localhost/v1/models

# Chat UI
curl -k https://localhost/
```

### 4. Managing Caddy

```bash
# Reload config without restarting
docker exec caddy caddy reload --config /etc/caddy/Caddyfile

# View logs
docker logs caddy

# Stop
docker stop caddy && docker rm caddy
```

---

## Comparison

| | nginx | Caddy |
|---|---|---|
| TLS setup | Manual (openssl + cert files) | Automatic (`tls internal`) |
| Config syntax | Verbose but familiar | Minimal |
| IP restriction | `allow` / `deny` directives | `remote_ip` matcher |
| Image size | ~40 MB | ~45 MB |
| Community | Ubiquitous, massive docs | Smaller, modern |
| Recommended if | Already using nginx elsewhere | Want simplest setup |

---

## Alternative: SSH Tunnel (No Proxy Needed)

If you only need admin access to the chat UI from your workstation and don't need HTTPS for the API, skip the reverse proxy entirely:

```bash
# From your workstation
ssh -L 3000:localhost:3000 -L 8000:localhost:8000 biscloud@<gb10-ip>
```

Then open `http://localhost:3000` in your browser. All traffic tunnels through SSH (already encrypted). No certs, no proxy, no config files.

| Port | Forwards to |
|---|---|
| `localhost:3000` | Open WebUI on GB10 |
| `localhost:8000` | vLLM API on GB10 |

This is the simplest option if you're always SSH'd into the box anyway.

---

## Open WebUI Backend Connection

Regardless of which proxy you choose, Open WebUI's internal connection to vLLM stays on plain HTTP (traffic never leaves the host):

- **Admin > Settings > Connections > OpenAI API Base URL:** `http://172.17.0.1:8000/v1`
- The HTTPS layer is only for external clients connecting through the reverse proxy

---

## Adding More Allowed IPs

To allow additional admin IPs without restarting:

**nginx:** Edit `~/nginx/nginx.conf`, add another `allow <ip>;` line, then:
```bash
docker exec nginx nginx -s reload
```

**Caddy:** Edit `~/caddy/Caddyfile`, add the IP to the `remote_ip` list, then:
```bash
docker exec caddy caddy reload --config /etc/caddy/Caddyfile
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `curl: (7) Failed to connect` | Proxy container not running or port conflict | `docker ps` to check; ensure nothing else on :443 |
| `502 Bad Gateway` | vLLM or Open WebUI down | Check `docker ps` for both backend containers |
| `403 Forbidden` on chat UI | Your IP not in allow list | Add your IP to the config and reload |
| Browser cert warning | Self-signed cert (expected) | Click through, or import Caddy's root CA |
| Chat UI loads but no LLM responses | Open WebUI backend URL wrong | Set to `http://172.17.0.1:8000/v1` in admin settings |
| API timeout on long prompts | `proxy_read_timeout` too low | Increase to 300s+ in nginx config |
