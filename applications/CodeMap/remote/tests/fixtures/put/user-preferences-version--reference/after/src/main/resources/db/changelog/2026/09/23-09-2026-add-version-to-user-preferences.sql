--liquibase formatted sql
--changeset system:add-version-to-user-preferences

-- Optimistic locking for user preferences: the @Version column starts at 0 for existing rows.
ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 0;

-- rollback ALTER TABLE user_preferences DROP COLUMN version;
