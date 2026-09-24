--liquibase formatted sql
--changeset norbert:add-version-to-user-preferences

-- Add JPA @Version column to user_preferences for optimistic locking protection
-- Prevents the web app and the mobile app from silently overwriting each other's changes
ALTER TABLE public.user_preferences
ADD COLUMN version BIGINT DEFAULT 0 NOT NULL;

-- rollback ALTER TABLE public.user_preferences DROP COLUMN version;
