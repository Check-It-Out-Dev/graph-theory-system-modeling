--liquibase formatted sql
--changeset system:add-display-order-to-currency

-- A display order chosen by the team; PLN first, the rest keep 0 until the team orders them.
ALTER TABLE currency ADD COLUMN IF NOT EXISTS display_order INT NOT NULL DEFAULT 0;

UPDATE currency SET display_order = 1 WHERE iso_code = 'PLN';

-- rollback ALTER TABLE currency DROP COLUMN display_order;
