# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it by creating a new issue with the "security" label or by contacting the repository owner directly.

## Security Updates

### April 2025: CVE-2025-43859 - HTTP Request Smuggling in h11 Python Library

**Vulnerability**: Prior to version 0.16.0, h11 had a leniency in parsing line terminators in chunked-coding message bodies that could lead to request smuggling vulnerabilities.

**Impact**: This vulnerability could potentially allow attackers to perform HTTP request smuggling attacks, which might lead to:
- Bypassing security controls
- Gaining unauthorized access to protected resources
- Stealing user credentials or session tokens

**Resolution**: Updated h11 dependency to version 0.16.0 which patches this vulnerability.

**Details**: 
- CVE: [CVE-2025-43859](https://nvd.nist.gov/vuln/detail/CVE-2025-43859)
- GitHub Security Advisory: [GHSA-vqfr-h8mv-ghfj](https://github.com/python-hyper/h11/security/advisories/GHSA-vqfr-h8mv-ghfj)

**Affected Versions**: h11 0.15.0 and earlier
**Patched Version**: h11 0.16.0

This update was implemented in PR [#XX] (to be filled after PR creation).
