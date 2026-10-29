variable "project_id" {
  description = "Google Cloud Project ID hosting AlphaScout"
  type        = string
}

variable "region" {
  description = "Primary Google Cloud region for Cloud Run and Vertex AI"
  type        = string
  default     = "us-central1"
}

variable "container_image" {
  description = "Artifact Registry container image URI for AlphaScout ADK service"
  type        = string
  default     = "us-central1-docker.pkg.dev/my-project/alphascout/agent:latest"
}

variable "vertex_search_datastore_id" {
  description = "Vertex AI Search DataStore ID for SEC 10-K/10-Q filings"
  type        = string
  default     = "alphascout-sec-filings-datastore"
}

variable "alpaca_api_key_secret_id" {
  description = "Secret Manager ID for Alpaca Paper Brokerage API Key"
  type        = string
  default     = "alphascout-alpaca-paper-api-key"
}

variable "alpaca_secret_key_secret_id" {
  description = "Secret Manager ID for Alpaca Paper Brokerage Secret Key"
  type        = string
  default     = "alphascout-alpaca-paper-secret-key"
}
