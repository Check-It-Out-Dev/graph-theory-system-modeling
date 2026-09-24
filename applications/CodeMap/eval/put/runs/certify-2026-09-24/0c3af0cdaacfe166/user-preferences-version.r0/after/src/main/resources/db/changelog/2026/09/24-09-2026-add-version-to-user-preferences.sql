-- liquibase formatted sql
-- changeset norbert:add-version-to-user-preferences-table

-- Add JPA @Version column to user_preferences table for optimistic locking protection
-- Prevents the web app and the mobile app from silently overwriting each other's
-- concurrent preference updates
ALTER TABLE user_preferences
ADD COLUMN version BIGINT DEFAULT 0 NOT NULL;

-- rollback ALTER TABLE user_preferences DROP COLUMN version;
