# Object Storage Setup for Training Data

**Problem:** Git cannot handle large checkpoint files (>100MB). Training artifacts (`.npz` files) were failing to sync.

**Solution:** Scaleway Object Storage for training data + Git for code.

---

## Quick Start

### 1. Get Scaleway IAM Credentials

1. Go to: https://console.scaleway.com/iam/api-keys
2. Click **"Create an access key"**
3. Name it: `gibbonbot-training`
4. Copy **Access Key** and **Secret Key** (save these securely!)

> ⚠️ **These keys are only shown once!** Save them in a password manager.

### 2. Configure rclone

```bash
rclone config
```

Answer prompts:
```
n) New remote
name> scaleway
Storage> s3
provider> 17 (Scaleway)
env_auth> false
access_key_id> [PASTE YOUR ACCESS KEY]
secret_access_key> [PASTE YOUR SECRET KEY]
region> 2 (Paris, France = fr-par)
endpoint> 2 (s3.fr-par.scw.cloud)
acl> 1 (private)
storage_class> 2 (STANDARD)
Edit advanced> n
y) Yes
```

### 3. Test Connection

```bash
rclone ls scaleway:gibbonbot-training/
```

---

## General Rules: What to Upload Where

### ✅ UPLOAD TO OBJECT STORAGE

| File Pattern | Reason |
|--------------|--------|
| `*.npz` | Checkpoints and rollouts (50-100MB each, git can't handle) |
| `mlflow.db` | SQLite database (grows with training metadata) |
| `mlruns/` | MLflow artifacts directory |
| `train.log` | Training logs |
| `lineage.json` | Run lineage/provenance tracking |
| `keepers/*.npz` | Curated checkpoints |
| `keepers/*.json` | Checkpoint metadata |

**Command:**
```bash
# Upload everything that belongs in object storage
rclone copy runs/ mlflow.db mlruns/ scaleway:gibbonbot-training/ -P
```

### ❌ KEEP IN GIT (NOT Object Storage)

| File Pattern | Reason |
|--------------|--------|
| `*.py` | Source code |
| `*.md` | Documentation |
| `*.txt` | Config, notes |
| `*.yml` / `*.yaml` | Configuration files |
| `*.json` (except lineage) | Small config files |
| `Dockerfile` | Container definition |
| `requirements.txt` | Dependencies |
| `.gitignore` | Git config |

### 📁 RECOMMENDED .gitignore ADDITIONS

```gitignore
# Training artifacts (go to object storage)
*.npz
mlflow.db
mlruns/
*.db
rollouts/
_train.log
lineage.json
keepers/
```

### 🔄 SYNC RULES SUMMARY

| Scenario | Action |
|----------|--------|
| New checkpoint saved | `rclone copy runs/ scaleway:.../runs/ -P` |
| New MLflow run | `rclone copy mlruns/ mlflow.db scaleway:.../ -P` |
| Teammate needs data | They run `rclone sync scaleway:.../runs/ runs/` |
| Code change | `git add/commit/push` (normal git flow) |

---

## Data Organization

| Data Type | Storage | Location |
|-----------|---------|----------|
| Code | Git | `git push` |
| Checkpoints `.npz` | Object Storage | `scaleway:gibbonbot-training/runs/` |
| Rollouts `.npz` | Object Storage | `scaleway:gibbonbot-training/runs/` |
| MLflow metadata | Object Storage | `scaleway:gibbonbot-training/mlruns/` |
| SQLite DB | Object Storage | `scaleway:gibbonbot-training/mlflow.db` |

---

## Sync Commands

### Upload Training Data

```bash
# From project root
rclone copy runs/ scaleway:gibbonbot-training/runs/ -P \
    --include "*.npz" \
    --include "*.json" \
    --include "keepers/**" \
    --include "train.log"

# MLflow data
rclone copy mlflow.db scaleway:gibbonbot-training/ -P
rclone copy mlruns/ scaleway:gibbonbot-training/mlruns/ -P
```

### Download Training Data

```bash
# Pull checkpoints
rclone sync scaleway:gibbonbot-training/runs/ape_walk_mjx/ runs/ape_walk_mjx/ -P

# Pull MLflow
rclone sync scaleway:gibbonbot-training/mlruns/ mlruns/ -P
rclone sync scaleway:gibbonbot-training/mlflow.db . -P
```

### Continuous Sync (Pull Every 30s)

```bash
# Save as scripts/pull_object_storage.sh
while true; do
    rclone sync scaleway:gibbonbot-training/runs/ape_walk_mjx/ runs/ape_walk_mjx/ -P
    rclone sync scaleway:gibbonbot-training/mlruns/ mlruns/ -P
    rclone sync scaleway:gibbonbot-training/mlflow.db . -P
    sleep 30
done
```

Run it:
```bash
chmod +x scripts/pull_object_storage.sh
./scripts/pull_object_storage.sh
```

---

## Modified Training Scripts

### `scripts/sync.sh` (Replace Git Push)

```bash
#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

echo "[sync] Uploading to Scaleway Object Storage..."

# Checkpoints
rclone copy runs/ scaleway:gibbonbot-training/runs/ -P \
    --include "*.npz" \
    --include "*.json" \
    --include "keepers/**" \
    --include "train.log"

# MLflow
rclone copy mlruns/ scaleway:gibbonbot-training/mlruns/ -P
rclone copy mlflow.db scaleway:gibbonbot-training/ -P

echo "[sync] Done!"
```

### `remote.sh` (For H100 Cloud Training)

Add upload after training in the `train_cmd`:
```bash
# Inside the train_cmd string, add this after the python command:
; rclone copy $RUN_DIR/ scaleway:gibbonbot-training/$RUN_DIR/ -P \
    --include "*.npz" --include "*.json" --include "train.log"
```

---

## Team Collaboration

### For Your Teammate

1. **Same process:** They install rclone and configure with same credentials
2. **Or separate IAM user:** Create a new IAM user for them in Scaleway
3. **Access:** They can now run the same sync commands

### Share a File Temporarily

```bash
# Generate 1-hour URL
rclone link scaleway:gibbonbot-training/runs/ape_walk_mjx/mjx_00012345.npz
```

---

## Troubleshooting

### "Access Denied" Error

1. Verify your access key is correct
2. Check the secret key
3. Ensure the IAM user has `ObjectStorage Read/Write` permissions
4. Confirm the region is `fr-par`

### Slow Uploads

- Use `-P` flag for progress
- Large files upload in chunks automatically
- Network latency is normal for first upload

### Verify Credentials

```bash
rclone config show scaleway:
```

---

## Costs

| Item | Cost (May 2025) |
|------|-----------------|
| Storage | €5.99/TB/month |
| Upload | Free |
| Download | €0.02/GB |
| Requests | ~€0.004 per 1000 |

**Example:** 100GB storage + 10GB download = **~€0.80/month**

> ✅ **Free Trial:** 750GB free for 90 days (until Aug 9, 2026)

---

## Security Best Practices

1. **Never commit keys to git**
2. Use `.env` files for credentials (add to `.gitignore`)
3. Rotate keys every 90 days
4. Use IAM policies to restrict access per user
5. Keep bucket **private** (default)

---

## Bucket Info

- **Bucket Name:** `gibbonbot-training`
- **Region:** `fr-par` (Paris)
- **Endpoint:** `https://gibbonbot-training.s3.fr-par.scw.cloud`
- **Provider:** Scaleway Object Storage

---

## Why Not Git?

| Problem | Git | Object Storage |
|---------|-----|-----------------|
| Large files (>100MB) | ❌ Fails/slow | ✅ Handles easily |
| Team sharing | ⚠️ Requires push/pull | ✅ Direct access |
| Versioning | ✅ Yes | ✅ Optional |
| Cost | Free | ~€0.60/month |

---

## Useful Commands Reference

| Action | Command |
|--------|---------|
| List files | `rclone ls scaleway:gibbonbot-training/` |
| Upload file | `rclone copy file.npz scaleway:gibbonbot-training/runs/` |
| Download file | `rclone copy scaleway:gibbonbot-training/runs/file.npz .` |
| Sync directory | `rclone sync local/ scaleway:gibbonbot-training/remote/` |
| Delete file | `rclone delete scaleway:gibbonbot-training/runs/file.npz` |
| Check size | `rclone size scaleway:gibbonbot-training/` |
| Show config | `rclone config show scaleway:` |
