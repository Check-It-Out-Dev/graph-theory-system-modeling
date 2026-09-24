--liquibase formatted sql
--changeset norbert:add-version-to-user-preferences-table

-- Add JPA @Version column to user_preferences table for optimistic locking protection
-- Prevents lost updates when the web app and mobile app concurrently edit the same preferences
ALTER TABLE user_preferences
ADD COLUMN version BIGINT DEFAULT 0 NOT NULL;

-- rollback ALTER TABLE user_preferences DROP COLUMN version;
