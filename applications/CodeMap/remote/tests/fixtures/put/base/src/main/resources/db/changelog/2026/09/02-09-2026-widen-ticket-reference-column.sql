-- liquibase formatted sql
-- changeset system:widen-ticket-reference-column

-- Widen ticket_reference from VARCHAR(20) to VARCHAR(32).
-- The pentest 3.3 fix (SecureRandom, Crockford base32) grew the reference
-- format from CIO-yyyyMMdd-XXXX (17 chars) to CIO-yyyyMMdd-XXXXXXXX
-- (21 chars), which exceeded the old column width. Widening VARCHAR is a
-- metadata-only ALTER in PostgreSQL — instant, no table rewrite, and all
-- existing 17-char references remain valid.

ALTER TABLE support_ticket ALTER COLUMN ticket_reference TYPE VARCHAR(32);

-- rollback ALTER TABLE support_ticket ALTER COLUMN ticket_reference TYPE VARCHAR(20);
