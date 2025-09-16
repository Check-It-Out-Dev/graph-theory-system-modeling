# Legal Compliance and Tool Usage

## Executive Summary

This project is **100% legally compliant** for internal team use. We use open-source tools within their license terms and clearly separate our methodology (MIT licensed) from the tools themselves.

## Tool-by-Tool Compliance

### Neo4j Community Edition

**License**: GPLv3  
**Our Usage**: ✅ **Fully Compliant**

- **What we do**: Use Neo4j Community Edition as an internal developer tool
- **What GPLv3 allows**: Unlimited internal use without any restrictions
- **What we DON'T do**: 
  - We don't distribute Neo4j
  - We don't sell Neo4j as a service
  - We don't modify Neo4j source code
- **Key Point**: The GPLv3 license does NOT affect:
  - Data stored in Neo4j
  - Applications connecting via REST/Bolt APIs
  - Our methodology or research papers

**Official Neo4j Statement**: "If you're using Neo4j internally at your organization and not distributing it, you can use Community Edition freely."

### Vector Embeddings

**Our Approach**: ✅ **Fully Compliant**

We compute embeddings **separately** from Neo4j using:

1. **Open-source models** (Apache 2.0/MIT licensed):
   - Sentence-Transformers
   - CodeBERT
   - All-MiniLM-L6-v2

2. **Optional commercial APIs** (properly licensed):
   - OpenAI Embeddings API (paid per use)
   - Other commercial services (with appropriate licenses)

**Key Point**: Embeddings are computed externally and stored as simple properties in Neo4j, maintaining complete separation of concerns.

### Development Process Timeline

**Initial Research Phase (Completed - Single User)**:
- **Who**: Norbert Marchewka (CheckItOut architect) only
- **What**: Neo4j Desktop Enterprise Edition with native embeddings
- **Where**: Personal computer, not shared with team
- **Why**: Initial discovery and pattern validation
- **Compliance**: Fully allowed under Neo4j evaluation terms
- **Current Status**: Research complete, preparing migration

**Team Deployment Phase (In Progress - Multi User)**:
- **Who**: Entire development team
- **What**: Neo4j Community Edition with SEPARATE embedding service
- **Where**: Shared team infrastructure
- **Why**: Team-wide adoption of discovered patterns
- **Key Change**: Embeddings moved from native Neo4j to external service
- **Compliance**: GPLv3 compliant for internal use
- **Current Status**: Migration from Enterprise to Community underway

**Critical Distinction**:
- Enterprise native embeddings were ONLY used during personal research
- Team deployment will NOT use any Enterprise features
- Embeddings are now computed OUTSIDE Neo4j for legal compliance

## Frequently Asked Questions

### Q: Can we use this in our company?
**A**: Yes, absolutely. The methodology is MIT licensed, and Neo4j Community Edition explicitly allows internal corporate use.

### Q: Do we need to open-source our code?
**A**: No. The GPLv3 license of Neo4j only applies if you distribute Neo4j itself. Using it internally or via APIs doesn't trigger GPL requirements.

### Q: Can we use this for commercial products?
**A**: Yes, with caveats:
- Our methodology (MIT licensed): Yes, freely
- Neo4j Community: Yes, for internal tools
- If building a product to sell: Consider Neo4j commercial license

### Q: What about the AI tools mentioned?
**A**: 
- Claude (Anthropic): Standard commercial service, pay per use
- Local LLMs: Various open-source licenses, check each model
- Our usage descriptions: Purely informational

### Q: Can Neo4j or Anthropic sue us?
**A**: No. We're using their tools exactly as intended:
- Neo4j encourages Community Edition for internal tools
- Anthropic welcomes research papers mentioning Claude
- We're customers, not competitors

## Best Practices for Compliance

1. **Be Transparent**: Always mention you're using Neo4j Community Edition as an internal tool
2. **Separate Concerns**: Keep embeddings computation separate from Neo4j
3. **Document Usage**: Maintain clear records of which version you're using
4. **Check Model Licenses**: When using new embedding models, verify their licenses
5. **Use APIs**: Connect to Neo4j via REST/Bolt APIs, not by embedding the database

## Legal Safety Checklist

- [x] Neo4j Community Edition for internal use only
- [x] No distribution of modified Neo4j code
- [x] Vector embeddings computed separately
- [x] Clear separation between our code (MIT) and tools (various licenses)
- [x] Proper attribution in documentation
- [x] No proprietary code from Neo4j or Anthropic included
- [x] Research papers describe usage, not redistribution

## Contact for Legal Questions

If you have specific legal concerns about implementing this methodology in your organization:

1. Consult your company's legal department
2. Review Neo4j's official licensing FAQ: https://neo4j.com/licensing/
3. For embedding models, check each model's specific license
4. This document provides information, not legal advice

## Summary

This project demonstrates a methodology using legally available tools:

**Research Phase** (Norbert Marchewka only):
- ✅ Neo4j Desktop Enterprise (evaluation license, single user)
- ✅ Native embeddings (Enterprise feature, for research only)
- ✅ Not shared with team or deployed

**Production Phase** (Team-wide):
- ✅ Neo4j Community Edition (GPLv3, internal use)
- ✅ External embedding service (complete separation)
- ✅ No Enterprise features used or required

**We maintain**:
- Full respect for all licenses
- Clear separation between research and production
- Transparent documentation of our migration
- Complete legal compliance at every stage

You can confidently use this approach in your organization while maintaining full legal compliance.

---

*Last updated: September 2025*  
*This document provides information about license compliance but does not constitute legal advice.*
