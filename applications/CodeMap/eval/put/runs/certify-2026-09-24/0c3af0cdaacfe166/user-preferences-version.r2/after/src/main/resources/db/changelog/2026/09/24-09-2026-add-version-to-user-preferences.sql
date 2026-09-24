-- Liquibase formatted SQL
-- changeset norbert:add-version-to-user-preferences-table

-- Add JPA @Version column to user_preferences table for optimistic locking protection
-- Prevents data loss when web and mobile clients concurrently update the same preferences
ALTER TABLE user_preferences
ADD COLUMN version BIGINT DEFAULT 0 NOT NULL;

-- Rollback
--rollback ALTER TABLE user_preferences DROP COLUMN version;
