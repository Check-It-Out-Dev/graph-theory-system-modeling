-- Liquibase formatted SQL
-- changeset norbert:add-version-to-support-ticket-table

-- Add JPA @Version column to support_ticket table for optimistic locking protection
-- Prevents lost updates when an admin and the customer modify the same ticket concurrently
ALTER TABLE support_ticket
ADD COLUMN version BIGINT DEFAULT 0 NOT NULL;

-- Rollback
--rollback ALTER TABLE support_ticket DROP COLUMN version;
