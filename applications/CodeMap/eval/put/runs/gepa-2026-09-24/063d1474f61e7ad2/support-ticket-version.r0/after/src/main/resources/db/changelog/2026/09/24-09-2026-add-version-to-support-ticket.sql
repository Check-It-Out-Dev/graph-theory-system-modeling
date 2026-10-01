-- liquibase formatted sql
-- changeset norbert:add-version-to-support-ticket

-- Add JPA @Version column to support_ticket for optimistic locking.
-- Prevents lost updates when an admin status change and a customer reply
-- happen at the same time.
ALTER TABLE support_ticket ADD COLUMN version BIGINT DEFAULT 0 NOT NULL;

-- rollback ALTER TABLE support_ticket DROP COLUMN version;
