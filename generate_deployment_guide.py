"""Generate DCI-VTON AWS Deployment Guide PDF."""
from fpdf import FPDF

class DeploymentPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "DCI-VTON AWS Deployment Guide | Confidential", align="R", new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(0, 102, 51)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def code_block(self, code):
        self.set_font("Courier", "", 9)
        self.set_fill_color(240, 240, 240)
        self.set_text_color(0, 0, 0)
        x = self.get_x()
        y = self.get_y()
        lines = code.strip().split("\n")
        block_h = len(lines) * 5 + 6
        if y + block_h > 270:
            self.add_page()
        self.rect(10, self.get_y(), 190, block_h, "F")
        self.ln(3)
        for line in lines:
            self.cell(5)
            self.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(0, 0, 0)
        self.cell(8)
        self.cell(5, 5.5, "-")
        self.multi_cell(170, 5.5, text)
        self.ln(1)

    def warning_box(self, text):
        self.set_fill_color(255, 245, 230)
        self.set_draw_color(255, 165, 0)
        y = self.get_y()
        self.rect(10, y, 190, 12, "DF")
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(200, 100, 0)
        self.cell(5)
        self.cell(0, 12, "WARNING: " + text, new_x="LMARGIN", new_y="NEXT")
        self.ln(3)


pdf = DeploymentPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ─── COVER PAGE ─────────────────────────────────────────────────────
pdf.add_page()
pdf.ln(50)
pdf.set_font("Helvetica", "B", 28)
pdf.set_text_color(0, 102, 51)
pdf.cell(0, 15, "DCI-VTON", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "B", 18)
pdf.set_text_color(0, 0, 0)
pdf.cell(0, 12, "AWS GPU Deployment Guide", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(10)
pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 8, "Virtual Try-On System - Production Deployment", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "Version 1.0 | June 2026", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(30)
pdf.set_font("Helvetica", "B", 11)
pdf.set_text_color(0, 0, 0)
pdf.cell(0, 8, "Architecture: FastAPI + Celery + GPU Inference (T4/A10G)", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "Target Inference Time: < 30 seconds per job", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 8, "GPU VRAM Required: ~5GB (T4 16GB recommended)", align="C", new_x="LMARGIN", new_y="NEXT")

# ─── TABLE OF CONTENTS ──────────────────────────────────────────────
pdf.add_page()
pdf.section_title("Table of Contents")
toc = [
    ("1.", "Architecture Overview", 3),
    ("2.", "Prerequisites & AWS Account Setup", 4),
    ("3.", "GPU Instance Selection & Launch", 5),
    ("4.", "Server Environment Setup", 7),
    ("5.", "Model Weights Download", 8),
    ("6.", "Project Code Deployment", 9),
    ("7.", "Environment Configuration", 10),
    ("8.", "Docker Build & Launch", 11),
    ("9.", "Testing & Verification", 12),
    ("10.", "Domain & SSL Setup", 13),
    ("11.", "Cost Optimization", 14),
    ("12.", "Monitoring & Maintenance", 15),
    ("13.", "Troubleshooting", 16),
]
for num, title, page in toc:
    pdf.set_font("Helvetica", "B" if num.startswith(("1.","2.","3.")) else "", 11)
    pdf.cell(10, 7, num)
    pdf.cell(140, 7, title)
    pdf.cell(0, 7, str(page), align="R", new_x="LMARGIN", new_y="NEXT")

# ─── 1. ARCHITECTURE ────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("1. Architecture Overview")

pdf.sub_title("1.1 System Architecture")
pdf.body_text("The DCI-VTON system consists of the following components running on a single AWS GPU instance:")
pdf.bullet("Frontend (Next.js) - Port 3000 - User uploads person + garment images")
pdf.bullet("API Server (FastAPI) - Port 8000 - Handles requests, stores images")
pdf.bullet("Task Queue (Celery + Redis) - Async job processing")
pdf.bullet("GPU Worker (Celery) - Runs DCI-VTON inference on GPU")
pdf.bullet("Database (PostgreSQL) - Job status tracking")
pdf.bullet("Flower (Optional) - Port 5555 - Celery monitoring dashboard")

pdf.sub_title("1.2 Inference Pipeline (per job)")
pdf.body_text("Each try-on job executes these ML models sequentially on the GPU:")
pdf.bullet("SegFormer - Clothing segmentation (~500MB, ~2 sec)")
pdf.bullet("DensePose - Body pose estimation (~250MB, ~3 sec)")
pdf.bullet("AFWM - Garment warping to body shape (~200MB, ~2 sec)")
pdf.bullet("DCI-VTON - Diffusion-based try-on generation (~3.5GB, ~15 sec)")
pdf.bullet("Post-processing - Color correction + compositing (~1 sec)")
pdf.body_text("Total: ~4.5GB VRAM, ~20-30 seconds per job. All models are preloaded at worker startup (one-time ~3 min), then each job only runs inference.")

pdf.sub_title("1.3 Request Flow")
pdf.code_block("""User Browser
    |
    v
Frontend (Next.js :3000)
    |
    v
API Server (FastAPI :8000) --> PostgreSQL (job tracking)
    |
    v
Redis (task queue)
    |
    v
GPU Worker (Celery) --> GPU (T4/A10G)
    |                     |
    |    SegFormer -> DensePose -> AFWM -> DCI-VTON -> Post-process
    |
    v
Result image saved --> API returns result URL""")

# ─── 2. PREREQUISITES ───────────────────────────────────────────────
pdf.add_page()
pdf.section_title("2. Prerequisites & AWS Account Setup")

pdf.sub_title("2.1 AWS Account")
pdf.bullet("Active AWS account with billing enabled")
pdf.bullet("IAM user with AdministratorAccess (or EC2 + VPC + Route53 permissions)")
pdf.bullet("AWS CLI v2 installed locally: https://aws.amazon.com/cli/")

pdf.sub_title("2.2 Configure AWS CLI")
pdf.code_block("""aws configure
  AWS Access Key ID: <your-access-key>
  AWS Secret Access Key: <your-secret-key>
  Default region: ap-south-1  (or your preferred region)
  Default output format: json""")

pdf.sub_title("2.3 Request GPU Quota (CRITICAL - Do this FIRST)")
pdf.warning_box("GPU quota request can take 1-24 hours. Submit immediately before proceeding.")
pdf.body_text("AWS blocks GPU instance launches by default. You must request a quota increase:")
pdf.code_block("""1. Go to: AWS Console > Service Quotas > EC2
2. Search: "Running On-Demand G and VT instances"
3. Click "Request increase at account level"
4. New quota value: 4  (g4dn.xlarge needs 4 vCPUs)
5. Use case: "ML inference for virtual try-on application"
6. Submit and wait for approval (usually 1-24 hours)""")

pdf.sub_title("2.4 Set Billing Alert")
pdf.code_block("""1. AWS Console > Billing > Budgets > Create Budget
2. Budget type: Cost budget
3. Budget amount: $100/month
4. Alert threshold: 80%
5. Email notification: your-email@example.com""")

pdf.sub_title("2.5 Tools Required on Local Machine")
pdf.bullet("Git (for cloning repository)")
pdf.bullet("SSH client (built-in on Mac/Linux, PuTTY on Windows)")
pdf.bullet("SCP or rsync (for file upload to server)")

# ─── 3. GPU INSTANCE ────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("3. GPU Instance Selection & Launch")

pdf.sub_title("3.1 Recommended Instance Types")
pdf.body_text("Choose based on your budget and performance needs:")
pdf.code_block("""Instance        GPU         VRAM    vCPU  RAM    Cost/hr   Cost/mo(24x7)
----------      -----       ----    ----  ---    -------   ------------
g4dn.xlarge     T4          16GB    4     16GB   $0.526    ~$380
g5.xlarge       A10G        24GB    4     16GB   $1.006    ~$725
g4dn.2xlarge    T4          16GB    8     32GB   $0.752    ~$540

RECOMMENDED: g4dn.xlarge (same T4 GPU as Kaggle, cheapest option)
Our model needs ~5GB VRAM - T4's 16GB is more than enough.""")

pdf.sub_title("3.2 Create Key Pair")
pdf.code_block("""aws ec2 create-key-pair \\
    --key-name vton-key \\
    --query 'KeyMaterial' \\
    --output text > vton-key.pem

chmod 400 vton-key.pem    # Linux/Mac
# Windows: Properties > Security > Remove all except your user""")

pdf.sub_title("3.3 Create Security Group")
pdf.code_block("""# Create security group
aws ec2 create-security-group \\
    --group-name vton-sg \\
    --description "VTON application security group"

# Allow SSH (restrict to your IP for security)
aws ec2 authorize-security-group-ingress \\
    --group-name vton-sg --protocol tcp --port 22 \\
    --cidr $(curl -s ifconfig.me)/32

# Allow HTTP, HTTPS, API, Frontend
aws ec2 authorize-security-group-ingress \\
    --group-name vton-sg --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress \\
    --group-name vton-sg --protocol tcp --port 443 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress \\
    --group-name vton-sg --protocol tcp --port 8000 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress \\
    --group-name vton-sg --protocol tcp --port 3000 --cidr 0.0.0.0/0""")

pdf.sub_title("3.4 Launch EC2 Instance")
pdf.warning_box("Use Deep Learning AMI - it has CUDA, PyTorch, Docker pre-installed.")
pdf.code_block("""# Find latest Deep Learning AMI ID for your region
aws ec2 describe-images \\
    --owners amazon \\
    --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI*" \\
              "Name=architecture,Values=x86_64" \\
    --query 'Images | sort_by(@,&CreationDate) | [-1].ImageId' \\
    --output text

# Launch instance (replace AMI_ID with result above)
aws ec2 run-instances \\
    --image-id <AMI_ID> \\
    --instance-type g4dn.xlarge \\
    --key-name vton-key \\
    --security-groups vton-sg \\
    --block-device-mappings '[{
        "DeviceName": "/dev/sda1",
        "Ebs": {
            "VolumeSize": 150,
            "VolumeType": "gp3",
            "Iops": 3000,
            "Throughput": 125
        }
    }]' \\
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=vton-gpu}]'""")

pdf.add_page()
pdf.sub_title("3.5 Allocate Elastic IP (Fixed IP Address)")
pdf.code_block("""# Allocate
aws ec2 allocate-address --domain vpc

# Note the AllocationId from output, then associate:
aws ec2 associate-address \\
    --instance-id <INSTANCE_ID> \\
    --allocation-id <ALLOCATION_ID>

# Note the PUBLIC IP - this is your server's permanent address""")

# ─── 4. SERVER SETUP ────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("4. Server Environment Setup")

pdf.sub_title("4.1 SSH into Server")
pdf.code_block("""ssh -i vton-key.pem ubuntu@<ELASTIC_IP>""")

pdf.sub_title("4.2 Verify GPU")
pdf.code_block("""nvidia-smi

# Expected output should show:
# Tesla T4, 16GB VRAM, CUDA 12.x
# If GPU not visible, the AMI may need nvidia driver setup""")

pdf.sub_title("4.3 Install NVIDIA Container Toolkit (Docker GPU Support)")
pdf.code_block("""# Add NVIDIA repo
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \\
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \\
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU in Docker
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi""")

pdf.sub_title("4.4 Install Docker Compose v2")
pdf.code_block("""# Docker Compose plugin (v2)
sudo apt-get install docker-compose-plugin

# Verify
docker compose version""")

# ─── 5. MODEL WEIGHTS ───────────────────────────────────────────────
pdf.add_page()
pdf.section_title("5. Model Weights Download")
pdf.warning_box("Model weights are ~4GB total. Download directly on the server (faster than upload).")

pdf.sub_title("5.1 Create Weights Directory")
pdf.code_block("""mkdir -p /home/ubuntu/vton_project/ml/weights
cd /home/ubuntu/vton_project/ml/weights""")

pdf.sub_title("5.2 Download DCI-VTON Checkpoint (viton512.ckpt)")
pdf.body_text("This is the main diffusion model (~3.5GB). Download from the Kaggle dataset:")
pdf.code_block("""# Option A: Using Kaggle CLI
pip install kaggle
export KAGGLE_USERNAME=aisubscrip2
export KAGGLE_KEY=<your-kaggle-key>

kaggle datasets download aisubscrip2/dci-vton-weights \\
    --file viton512.ckpt \\
    -p /home/ubuntu/vton_project/ml/weights/ --force

# Unzip if downloaded as .zip
cd /home/ubuntu/vton_project/ml/weights/
unzip -o *.zip 2>/dev/null; rm -f *.zip

# Option B: Using direct URL (if hosted on S3/GDrive)
# wget <direct-url> -O viton512.ckpt""")

pdf.sub_title("5.3 Download AFWM Warp Weights (warp_viton.pth)")
pdf.code_block("""kaggle datasets download aisubscrip2/dci-vton-weights \\
    --file warp_viton.pth \\
    -p /home/ubuntu/vton_project/ml/weights/ --force

unzip -o *.zip 2>/dev/null; rm -f *.zip""")

pdf.sub_title("5.4 DensePose Weights (auto-downloaded)")
pdf.body_text("DensePose weights (~250MB) are automatically downloaded on first run. No manual action needed. The file will be saved to: /home/ubuntu/vton_project/ml/weights/densepose_rcnn_R_50_FPN_s1x.pkl")

pdf.sub_title("5.5 Verify All Weights")
pdf.code_block("""ls -lh /home/ubuntu/vton_project/ml/weights/

# Expected:
# viton512.ckpt           ~3.5GB  (DCI-VTON diffusion model)
# warp_viton.pth          ~200MB  (AFWM garment warp model)
# densepose_*.pkl         ~250MB  (auto-downloaded on first run)

# CRITICAL: Both viton512.ckpt and warp_viton.pth MUST exist
# If missing, the worker will fail to start""")

# ─── 6. PROJECT CODE ────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("6. Project Code Deployment")

pdf.sub_title("6.1 Clone Repository")
pdf.code_block("""cd /home/ubuntu
git clone <your-repo-url> vton_project
cd vton_project""")

pdf.sub_title("6.2 Project Structure (Key Files)")
pdf.code_block("""vton_project/
  api/
    Dockerfile.gpu              # GPU Docker image
    docker-compose.gpu.yml      # GPU docker-compose
    requirements.txt            # Base Python deps
    requirements-gpu.txt        # GPU Python deps (PyTorch, detectron2, etc.)
    app/
      main.py                   # FastAPI app
      config.py                 # Settings (reads .env)
      services/
        inference.py            # Inference router (auto-detects GPU mode)
      workers/
        tasks.py                # Celery task definitions
  ml/
    weights/                    # Model weights (Step 5)
      viton512.ckpt
      warp_viton.pth
    scripts/
      gpu_inference.py          # GPU inference engine
    kaggle/
      dci_vton_inference.ipynb  # Kaggle notebook (not used in GPU mode)
  storage/                      # Runtime storage (auto-created)
    inputs/                     # Uploaded images
    outputs/                    # Result images
  frontend/                     # Next.js frontend
  .env                          # Environment configuration""")

pdf.sub_title("6.3 Verify Weights are in Correct Location")
pdf.code_block("""# Weights must be at: vton_project/ml/weights/
ls -lh /home/ubuntu/vton_project/ml/weights/viton512.ckpt
ls -lh /home/ubuntu/vton_project/ml/weights/warp_viton.pth

# If weights were downloaded elsewhere, move them:
mv /path/to/viton512.ckpt /home/ubuntu/vton_project/ml/weights/
mv /path/to/warp_viton.pth /home/ubuntu/vton_project/ml/weights/""")

# ─── 7. ENVIRONMENT CONFIG ──────────────────────────────────────────
pdf.add_page()
pdf.section_title("7. Environment Configuration")

pdf.sub_title("7.1 Edit .env File")
pdf.code_block("""cd /home/ubuntu/vton_project
nano .env""")

pdf.sub_title("7.2 Required .env Settings for GPU Mode")
pdf.code_block("""# ---- Database ----
DATABASE_URL=postgresql+asyncpg://vton:vton@postgres:5432/vton

# ---- Celery / Redis ----
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# ---- Storage ----
STORAGE_BACKEND=local
LOCAL_STORAGE_PATH=/app/storage

# ---- ML / Inference (GPU MODE) ----
USE_OWN_MODEL=false
MODEL_PATH=
DEVICE=cuda
WEIGHTS_DIR=/app/ml/weights

# ---- Quality threshold ----
MIN_QUALITY_SCORE=0.65
TRAINING_PAIR_SSIM_THRESHOLD=0.65

# ---- API ----
MAX_UPLOAD_SIZE_MB=10

# ---- Kaggle (NOT NEEDED for GPU mode, can leave empty) ----
KAGGLE_USERNAME=
KAGGLE_KEY=
KAGGLE_NOTEBOOK_SLUG=
KAGGLE_DATASET_SLUG=""")

pdf.warning_box("DEVICE=cuda and WEIGHTS_DIR=/app/ml/weights are CRITICAL. Without these, GPU mode won't activate.")

pdf.sub_title("7.3 How Mode Detection Works")
pdf.body_text("The inference router auto-detects the mode based on .env settings:")
pdf.code_block("""Priority order:
1. DEVICE=cuda + WEIGHTS_DIR set + weights exist  --> Local GPU mode
2. KAGGLE_USERNAME + KAGGLE_KEY set               --> Kaggle mode
3. Neither                                         --> Placeholder mode

GPU mode: ~20-30 sec per job (direct GPU inference)
Kaggle mode: ~5-10 min per job (remote notebook execution)
Placeholder mode: ~3 sec (fake result for testing)""")

# ─── 8. DOCKER BUILD ────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("8. Docker Build & Launch")

pdf.sub_title("8.1 Create Docker Network")
pdf.code_block("""docker network create vton_net 2>/dev/null || true""")

pdf.sub_title("8.2 Build and Start All Services")
pdf.code_block("""cd /home/ubuntu/vton_project/api

# Build (first time: ~15-20 min for GPU dependencies)
docker compose -f docker-compose.gpu.yml build

# Start all services
docker compose -f docker-compose.gpu.yml up -d

# Check all containers are running
docker compose -f docker-compose.gpu.yml ps""")

pdf.sub_title("8.3 Expected Output")
pdf.code_block("""NAME              STATUS          PORTS
api-api-1         Up              0.0.0.0:8000->8000/tcp
api-worker-1      Up              (GPU attached)
api-postgres-1    Up (healthy)    0.0.0.0:5433->5432/tcp
api-redis-1       Up (healthy)    0.0.0.0:6379->6379/tcp
api-flower-1      Up              0.0.0.0:5555->5555/tcp""")

pdf.sub_title("8.4 Check Worker GPU Access")
pdf.code_block("""# Verify GPU is accessible inside worker container
docker exec api-worker-1 nvidia-smi

# Check worker logs - should show "Local GPU mode active"
docker logs api-worker-1 --tail 30

# Expected log lines:
# InferenceRouter: Local GPU mode - loading models...
# SegFormer loaded.
# DensePose loaded.
# AFWM warp loaded.
# DCI-VTON model loaded.
# GPUInferenceEngine: all models loaded.
# InferenceRouter: Local GPU mode active.""")

pdf.sub_title("8.5 First Run Model Loading")
pdf.body_text("On first worker start, models take ~3-5 minutes to load into GPU memory. Subsequent job requests will be fast (~20-30 sec). DensePose weights will auto-download on first run (~250MB).")

# ─── 9. TESTING ─────────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("9. Testing & Verification")

pdf.sub_title("9.1 Health Check")
pdf.code_block("""curl http://localhost:8000/docs
# Should return FastAPI Swagger UI HTML""")

pdf.sub_title("9.2 Submit Test Job via curl")
pdf.code_block("""# Upload person + garment images and submit try-on job
curl -X POST http://localhost:8000/api/tryon \\
    -F "person_image=@/path/to/test_person.jpg" \\
    -F "garment_image=@/path/to/test_garment.jpg"

# Response:
# {"job_id": "abc-123-def", "status": "pending", "eta_seconds": 35}""")

pdf.sub_title("9.3 Check Job Status")
pdf.code_block("""curl http://localhost:8000/api/tryon/<job_id>

# Wait 20-30 seconds, then check again
# When status="completed", result_image_path will have the output""")

pdf.sub_title("9.4 Check Worker Logs")
pdf.code_block("""docker logs api-worker-1 --tail 50 -f

# Watch for:
# [INFO] SegFormer loaded
# [INFO] DCI inference complete!
# [INFO] Result saved: /app/storage/outputs/abc-123.jpg""")

pdf.sub_title("9.5 Performance Verification")
pdf.code_block("""# Expected timings per job:
# SegFormer segmentation:    ~2 sec
# DensePose extraction:      ~3 sec
# AFWM garment warping:      ~2 sec
# DCI-VTON diffusion:        ~15 sec
# Post-processing:           ~1 sec
# TOTAL:                     ~23 sec

# If inference takes >60 sec, check:
# 1. nvidia-smi (GPU utilization should be >80%)
# 2. docker stats (memory usage)
# 3. Worker logs for errors""")

# ─── 10. DOMAIN & SSL ───────────────────────────────────────────────
pdf.add_page()
pdf.section_title("10. Domain & SSL Setup (Optional)")

pdf.sub_title("10.1 Install Nginx")
pdf.code_block("""sudo apt-get update
sudo apt-get install -y nginx""")

pdf.sub_title("10.2 Configure Nginx")
pdf.code_block("""sudo nano /etc/nginx/sites-available/vton

# Paste this configuration:
server {
    listen 80;
    server_name api.yourdomain.com;

    client_max_body_size 20M;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 300s;
    }
}

server {
    listen 80;
    server_name app.yourdomain.com;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}""")

pdf.code_block("""# Enable site
sudo ln -s /etc/nginx/sites-available/vton /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx""")

pdf.sub_title("10.3 SSL with Let's Encrypt")
pdf.code_block("""sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com -d app.yourdomain.com

# Auto-renewal test
sudo certbot renew --dry-run""")

# ─── 11. COST OPTIMIZATION ──────────────────────────────────────────
pdf.add_page()
pdf.section_title("11. Cost Optimization")

pdf.sub_title("11.1 Cost Comparison")
pdf.code_block("""Option                  Cost/hour   Cost/month(24x7)   Best for
----                    ---------   ----------------   --------
On-Demand               $0.526      ~$380              Always-on production
Spot Instance           $0.158      ~$115              Dev/testing (can interrupt)
Reserved 1-year         $0.330      ~$240              Committed production
Reserved 3-year         $0.220      ~$160              Long-term production
Auto Start/Stop         varies      ~$50-100           Low traffic""")

pdf.sub_title("11.2 Auto Start/Stop (Recommended for Low Traffic)")
pdf.body_text("Create a Lambda function that stops the instance when no API requests for 30 minutes, and a simple health-check endpoint that starts the instance on demand.")

pdf.sub_title("11.3 Spot Instance (70% Cheaper)")
pdf.code_block("""aws ec2 request-spot-instances \\
    --instance-count 1 \\
    --type persistent \\
    --launch-specification '{
        "ImageId": "<AMI_ID>",
        "InstanceType": "g4dn.xlarge",
        "KeyName": "vton-key",
        "SecurityGroups": ["vton-sg"]
    }'

# WARNING: AWS can terminate spot instances with 2-min notice
# Use for development/testing only, not production""")

# ─── 12. MONITORING ─────────────────────────────────────────────────
pdf.add_page()
pdf.section_title("12. Monitoring & Maintenance")

pdf.sub_title("12.1 Docker Container Monitoring")
pdf.code_block("""# View all container status
docker compose -f docker-compose.gpu.yml ps

# View resource usage
docker stats

# View worker logs (live)
docker logs api-worker-1 -f --tail 100

# View API logs
docker logs api-api-1 -f --tail 100""")

pdf.sub_title("12.2 GPU Monitoring")
pdf.code_block("""# Real-time GPU usage
watch -n 1 nvidia-smi

# Inside worker container
docker exec api-worker-1 nvidia-smi""")

pdf.sub_title("12.3 Flower Dashboard (Celery Monitoring)")
pdf.body_text("Access at http://<ELASTIC_IP>:5555 - shows active tasks, worker status, and task history.")

pdf.sub_title("12.4 Restart Services")
pdf.code_block("""# Restart specific service
docker compose -f docker-compose.gpu.yml restart worker

# Restart all
docker compose -f docker-compose.gpu.yml restart

# Full rebuild (after code changes)
docker compose -f docker-compose.gpu.yml up --build -d""")

pdf.sub_title("12.5 Auto-Recovery")
pdf.code_block("""# Docker restart policy is set in docker-compose.gpu.yml
# Add to worker service if not present:
#   restart: unless-stopped

# EC2 auto-recovery:
aws ec2 modify-instance-attribute \\
    --instance-id <INSTANCE_ID> \\
    --auto-recovery-enabled""")

# ─── 13. TROUBLESHOOTING ────────────────────────────────────────────
pdf.add_page()
pdf.section_title("13. Troubleshooting")

pdf.sub_title("Problem: Worker fails to start")
pdf.code_block("""# Check logs
docker logs api-worker-1

# Common causes:
# 1. Weights missing: "FileNotFoundError: viton512.ckpt"
#    Fix: Download weights to ml/weights/ (see Step 5)
#
# 2. GPU not accessible: "CUDA not available"
#    Fix: Verify nvidia-container-toolkit installed (see Step 4.3)
#
# 3. Out of VRAM: "CUDA out of memory"
#    Fix: Use g4dn.xlarge (16GB) or reduce batch size""")

pdf.sub_title("Problem: Inference takes >60 seconds")
pdf.code_block("""# Check GPU utilization
docker exec api-worker-1 nvidia-smi

# If GPU util = 0%: model not on GPU
#   Check DEVICE=cuda in .env
#
# If GPU util = 100% but slow: thermal throttling
#   Check: nvidia-smi -q -d TEMPERATURE
#
# If GPU util normal but slow: CPU bottleneck
#   Check: docker stats (CPU usage)""")

pdf.sub_title("Problem: CUDA version mismatch")
pdf.code_block("""# Inside container:
docker exec api-worker-1 python -c "import torch; print(torch.version.cuda)"

# Must match the NVIDIA driver on host
nvidia-smi  # Shows driver CUDA version

# Fix: Use compatible PyTorch version in Dockerfile.gpu""")

pdf.sub_title("Problem: Port already in use")
pdf.code_block("""# Find what's using the port
sudo lsof -i :8000
sudo lsof -i :5432

# Kill the process or change port in docker-compose.gpu.yml""")

pdf.sub_title("Problem: Database connection refused")
pdf.code_block("""# Check PostgreSQL is healthy
docker compose -f docker-compose.gpu.yml ps postgres

# If unhealthy, check logs
docker logs api-postgres-1

# Reset database (WARNING: deletes all data)
docker compose -f docker-compose.gpu.yml down -v
docker compose -f docker-compose.gpu.yml up -d""")

# ─── QUICK REFERENCE CARD ───────────────────────────────────────────
pdf.add_page()
pdf.section_title("Quick Reference Card")

pdf.sub_title("Key Commands")
pdf.code_block("""# SSH into server
ssh -i vton-key.pem ubuntu@<ELASTIC_IP>

# Start services
cd /home/ubuntu/vton_project/api
docker compose -f docker-compose.gpu.yml up -d

# Stop services
docker compose -f docker-compose.gpu.yml down

# View logs
docker logs api-worker-1 -f

# Rebuild after code change
docker compose -f docker-compose.gpu.yml up --build -d

# Check GPU
nvidia-smi
docker exec api-worker-1 nvidia-smi

# Test API
curl -X POST http://<IP>:8000/api/tryon \\
    -F person_image=@person.jpg \\
    -F garment_image=@garment.jpg""")

pdf.sub_title("Key URLs")
pdf.code_block("""API:        http://<ELASTIC_IP>:8000
API Docs:   http://<ELASTIC_IP>:8000/docs
Frontend:   http://<ELASTIC_IP>:3000
Flower:     http://<ELASTIC_IP>:5555""")

pdf.sub_title("Key Files on Server")
pdf.code_block("""/home/ubuntu/vton_project/
    .env                            # Configuration
    api/docker-compose.gpu.yml      # GPU Docker compose
    ml/weights/viton512.ckpt        # DCI-VTON model (~3.5GB)
    ml/weights/warp_viton.pth       # AFWM warp model (~200MB)
    storage/outputs/                # Generated results""")

# Save
output_path = r"c:\vton_project\DCI-VTON_AWS_Deployment_Guide.pdf"
pdf.output(output_path)
print(f"PDF saved to: {output_path}")
