---
title: Integrations and API
doc_type: official
last_updated: 2026-06-05
---

# Integrations and API

## Available integrations

| Integration | Plans | What it does |
|---|---|---|
| Slack | All plans | Task notifications in channels, create tasks from messages |
| Microsoft Teams | All plans | Task notifications and a personal Brindlework tab |
| Google Drive / Dropbox | All plans | Attach cloud files to tasks |
| GitHub / GitLab | Starter and above | Link commits and pull requests to tasks |
| Webhooks | Starter and above | Send events to your own endpoints |
| Zapier | Pro and above | Connect Brindlework to 5,000+ apps |
| Salesforce | Enterprise only | Sync opportunities to projects |

Integrations are connected by a workspace Admin or Owner in Settings > Integrations. The GitLab integration was added in version 4.8.

## REST API

The Brindlework REST API is available on all plans. Create a personal API key in Settings > Developer > API keys. API keys inherit the permissions of the user who created them. Keys can be revoked at any time and expire automatically after 1 year.

For apps used by other people, register an OAuth 2.0 app in Settings > Developer > OAuth apps instead of sharing API keys.

## Rate limits

API rate limits are applied per workspace:

- Free: 60 requests per minute
- Starter: 300 requests per minute
- Pro: 1,000 requests per minute
- Enterprise: 5,000 requests per minute (higher limits available on request)

When a limit is exceeded, the API responds with HTTP status 429 and a `Retry-After` header that tells you how many seconds to wait before retrying.

## Webhooks

Webhooks send an HTTP POST request to your endpoint for events such as task.created, task.updated, and task.completed. Each request is signed with an HMAC-SHA256 signature in the `X-Brindlework-Signature` header. If your endpoint does not return a 2xx status, Brindlework retries up to 5 times with exponential backoff over about 1 hour.
