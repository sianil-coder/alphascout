#!/usr/bin/env bash
# ============================================================================
# One-Command Google Cloud Deployment & Verification Script for AlphaScout
# Target Project: sianiltestprj-sandbox-970332
# ============================================================================
set -euo pipefail

PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-sianiltestprj-sandbox-970332}"
REGION="${GOOGLE_CLOUD_LOCATION:-us-central1}"
SERVICE_NAME="alphascout-adk-agent"

echo "============================================================================"
echo " Deploying AlphaScout to GCP Project: ${PROJECT_ID} (${REGION})"
echo "============================================================================"

echo "[1/5] Setting active GCP project..."
gcloud config set project "${PROJECT_ID}"

echo "[2/5] Running local unit tests & 6-case Golden Dataset Evaluation first..."
export PYTHONPATH=.
python3 -m tests.eval.evaluate_agent
python3 -m unittest tests/test_alphascout.py -v

echo "[3/5] Enabling required Google Cloud APIs..."
gcloud services enable \
  run.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  aiplatform.googleapis.com \
  secretmanager.googleapis.com \
  discoveryengine.googleapis.com \
  dlp.googleapis.com \
  cloudtrace.googleapis.com \
  logging.googleapis.com \
  --project="${PROJECT_ID}"

echo "[4/5] Creating Artifact Registry repo, Building Image & Deploying to Cloud Run..."
AR_REPO="alphascout-repo"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${SERVICE_NAME}:latest"

gcloud artifacts repositories create "${AR_REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --project="${PROJECT_ID}" 2>/dev/null || true

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
docker build -t "${IMAGE_URI}" .
docker push "${IMAGE_URI}"

gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE_URI}" \
  --region "${REGION}" \
  --project "${PROJECT_ID}" \
  --ingress internal-and-cloud-load-balancing \
  --no-allow-unauthenticated \
  --set-env-vars="GOOGLE_GENAI_USE_VERTEXAI=TRUE,GOOGLE_CLOUD_PROJECT=${PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION}"

SERVICE_URL="$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --project "${PROJECT_ID}" --format='value(status.url)')"

echo "============================================================================"
echo "[5/5] Verifying Live Deployed Cloud Run Service at: ${SERVICE_URL}"
echo "============================================================================"
curl -sS "${SERVICE_URL}/healthz" | python3 -m json.tool

echo ""
echo "Running live Equity Research + HITL Gate smoke test against ${SERVICE_URL}/api/research ..."
curl -sS -X POST "${SERVICE_URL}/api/research" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"NVDA","peer_tickers":["AMD","AVGO"],"user_query":"Analyze NVDA for client account ACC-8841-9920 and stage a paper buy order.","human_approved_trade":false}' \
  | python3 -m json.tool

echo ""
echo "✅ Deployment & Live Smoke Test Complete!"
echo "🌐 Open your live AlphaScout Dashboard: ${SERVICE_URL}"
