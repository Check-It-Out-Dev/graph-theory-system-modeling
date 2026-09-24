--liquibase formatted sql
--changeset norbert_marchewka:add-version-to-support-ticket
ALTER TABLE support_ticket ADD COLUMN version BIGINT NOT NULL DEFAULT 0;
-- rollback ALTER TABLE support_ticket DROP COLUMN version;
