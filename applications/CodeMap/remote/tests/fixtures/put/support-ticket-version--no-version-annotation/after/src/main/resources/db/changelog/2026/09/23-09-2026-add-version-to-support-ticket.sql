--liquibase formatted sql
--changeset system:add-version-to-support-ticket

-- Optimistic locking for support tickets: the @Version column starts at 0 for existing rows.
ALTER TABLE support_ticket ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 0;

-- rollback ALTER TABLE support_ticket DROP COLUMN version;
