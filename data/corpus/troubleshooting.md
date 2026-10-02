---
title: Troubleshooting Guide
doc_type: official
last_updated: 2026-07-01
---

# Troubleshooting Guide

## Error codes

| Code | Meaning | What to do |
|---|---|---|
| E101 | Sync conflict: the task was edited on two devices at the same time | Open the task and choose which version to keep |
| E204 | Attachment is larger than your plan's file size limit | Compress the file or upgrade your plan |
| E301 | Permission denied | Ask a project Admin to give you access |
| E429 | Too many requests | Wait and retry; see API rate limits |
| E500 | Internal server error | Check status.brindlework.example and retry later |

## File size limits

The maximum size of a single attachment depends on the plan: Free 10 MB, Starter 100 MB, Pro 250 MB, and Enterprise 2 GB.

## Desktop app is not syncing

1. Check that you are online and that status.brindlework.example shows no incidents.
2. Sign out and sign back in.
3. Clear the local cache: open Help > Troubleshooting > Clear cache. This does not delete any data stored on Brindlework servers.
4. Update to the latest version. Desktop app versions older than 4.5 are no longer supported and cannot sync.

## Not receiving notifications

- Check Profile > Notifications to confirm the notification type is enabled.
- Email notifications are sent from notifications@brindlework.example; add this address to your allow list.
- Mobile push notifications require notification permission for the Brindlework app in your phone settings.
- If you use Do Not Disturb hours in Brindlework, notifications are held until the period ends.

## Cannot sign in

- If you use SSO, sign in through your company's identity provider instead of using a password.
- After 10 failed sign-in attempts, the account is locked for 15 minutes.
- If you lost your 2FA device, use one of your recovery codes.
