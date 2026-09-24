-- liquibase formatted sql
-- changeset norbert:add-version-to-support-ticket-table

-- Add JPA @Version column to support_ticket for optimistic locking.
-- Prevents an admin status change and a customer reply from silently
-- overwriting each other when they happen at the same moment.
ALTER TABLE support_ticket
ADD COLUMN version BIGINT DEFAULT 0 NOT NULL;

-- rollback ALTER TABLE support_ticket DROP COLUMN version;
