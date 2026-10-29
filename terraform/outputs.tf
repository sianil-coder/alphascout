output "cloud_run_service_url" {
  description = "Public HTTPS endpoint for the deployed AlphaScout ADK Service"
  value       = google_cloud_run_v2_service.alphascout_service.uri
}

output "agent_service_account_email" {
  description = "Least-privilege service account email for AlphaScout"
  value       = google_service_account.alphascout_agent_sa.email
}

output "vertex_search_datastore_id" {
  description = "Provisioned Vertex AI Search DataStore ID for SEC filings"
  value       = google_discovery_engine_data_store.sec_filings_datastore.data_store_id
}
