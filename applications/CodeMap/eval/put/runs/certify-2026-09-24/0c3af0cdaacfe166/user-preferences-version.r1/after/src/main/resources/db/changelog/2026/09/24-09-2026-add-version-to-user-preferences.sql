--liquibase formatted sql
--changeset system:add-version-to-user-preferences

-- Optimistic locking for user_preferences: web and mobile edited the same
-- row concurrently and one save silently overwrote the other's changes.
ALTER TABLE user_preferences ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL DEFAULT 0;

-- rollback ALTER TABLE user_preferences DROP COLUMN version;
