---
title: Account Security
doc_type: official
last_updated: 2026-04-22
---

# Account Security

## Passwords

Passwords must be at least 12 characters long and cannot match any of your last 5 passwords. Brindlework checks new passwords against known breached password lists. You can reset your password from the sign-in page using "Forgot password"; the reset link expires after 1 hour.

## Two-factor authentication (2FA)

Two-factor authentication is available to every user on every plan. Brindlework supports authenticator apps (TOTP) and hardware security keys (WebAuthn / FIDO2). SMS codes are not supported.

When you enable 2FA, you receive 10 single-use recovery codes. Store them safely: if you lose access to your authenticator and your recovery codes, a workspace Owner must reset your 2FA, or, if you are the only Owner, you must contact support and verify your identity.

Admins on Pro and Enterprise plans can require 2FA for all members in Settings > Security > Authentication. Members without 2FA are asked to set it up at their next sign-in.

## Single sign-on (SSO) and SCIM

SAML 2.0 single sign-on is available on the Enterprise plan only. Supported identity providers include Okta, Microsoft Entra ID (Azure AD), Google Workspace, and OneLogin. Enterprise workspaces can also use SCIM 2.0 to automatically provision and deprovision users.

## Sessions

By default, users stay signed in for 30 days. Enterprise admins can set a custom session length between 1 and 30 days. Admins on any paid plan can sign out a member from all devices in Settings > Members.

## Audit logs

Audit logs record sign-ins, permission changes, exports, and deletions.

- Pro: audit logs are kept for 90 days
- Enterprise: audit logs are kept for 1 year and can be streamed to a SIEM

Audit logs are not available on the Free and Starter plans.

## Encryption

All data is encrypted at rest with AES-256 and in transit with TLS 1.2 or higher. Brindlework is SOC 2 Type II certified. Security reports are available to Enterprise customers under NDA.

## Reporting a vulnerability

Report security issues to security@brindlework.example. Brindlework runs a responsible disclosure program and responds to reports within 3 business days.
