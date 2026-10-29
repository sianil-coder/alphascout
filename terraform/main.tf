# ============================================================================
# AlphaScout Infrastructure-as-Code (Terraform)
# Provisions Google Cloud production infrastructure:
# 1. Google Cloud Secret Manager secrets for Alpaca Paper Brokerage keys
# 2. Google Cloud Discovery Engine (Vertex AI Search) DataStore for SEC 10-K filings
# 3. Least-privilege Service Account & IAM bindings
# 4. Google Cloud Run v2 Service hosting the AlphaScout ADK Agent
# ============================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.30"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# 1. Enable Required Google Cloud APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "discoveryengine.googleapis.com",
    "secretmanager.googleapis.com",
    "dlp.googleapis.com",
    "run.googleapis.com",
    "cloudtrace.googleapis.com",
    "logging.googleapis.com",
  ])
  project            = var.project_id
  service            = each.key
  disable_on_destroy = false
}

# 2. Dedicated Service Account for AlphaScout ADK Agent
resource "google_service_account" "alphascout_agent_sa" {
  account_id   = "alphascout-agent-sa"
  display_name = "AlphaScout ADK Equity Research Agent Service Account"
}

resource "google_project_iam_member" "agent_iam_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/discoveryengine.viewer",
    "roles/secretmanager.secretAccessor",
    "roles/dlp.user",
    "roles/cloudtrace.agent",
    "roles/logging.logWriter",
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.alphascout_agent_sa.email}"
}

# 3. Google Cloud Secret Manager Secrets (No Hardcoded API Keys)
resource "google_secret_manager_secret" "alpaca_paper_api_key" {
  secret_id = var.alpaca_api_key_secret_id
  replication {
    auto {}
  }
  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret" "alpaca_paper_secret_key" {
  secret_id = var.alpaca_secret_key_secret_id
  replication {
    auto {}
  }
  depends_on = [google_project_service.required_apis]
}

# 4. Vertex AI Search DataStore for SEC 10-K / 10-Q Filings (Persistent Memory)
resource "google_discovery_engine_data_store" "sec_filings_datastore" {
  location                    = "global"
  data_store_id               = var.vertex_search_datastore_id
  display_name                = "AlphaScout SEC 10-K and 10-Q Filings Store"
  industry_vertical           = "GENERIC"
  content_config              = "CONTENT_REQUIRED"
  solution_types              = ["SOLUTION_TYPE_SEARCH"]
  create_advanced_site_search = false
  depends_on                  = [google_project_service.required_apis]
}

# 5. Cloud Run v2 Service for the AlphaScout ADK Agent
resource "google_cloud_run_v2_service" "alphascout_service" {
  name     = "alphascout-adk-agent"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_INTERNAL_LOAD_BALANCER"

  template {
    service_account = google_service_account.alphascout_agent_sa.email

    containers {
      image = var.container_image

      env {
        name  = "GOOGLE_GENAI_USE_VERTEXAI"
        value = "TRUE"
      }
      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "GOOGLE_CLOUD_LOCATION"
        value = var.region
      }
      env {
        name  = "VERTEX_SEARCH_DATASTORE_ID"
        value = google_discovery_engine_data_store.sec_filings_datastore.data_store_id
      }
      env {
        name  = "ALPACA_API_KEY_SECRET_ID"
        value = google_secret_manager_secret.alpaca_paper_api_key.secret_id
      }
      env {
        name  = "ALPACA_SECRET_KEY_SECRET_ID"
        value = google_secret_manager_secret.alpaca_paper_secret_key.secret_id
      }
    }
  }
  depends_on = [google_project_service.required_apis]
}
