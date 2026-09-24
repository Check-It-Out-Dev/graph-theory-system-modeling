--liquibase formatted sql
--changeset norbert:add-version-to-user-preferences
ALTER TABLE public.user_preferences ADD COLUMN version BIGINT NOT NULL DEFAULT 0;

-- rollback ALTER TABLE public.user_preferences DROP COLUMN version;
