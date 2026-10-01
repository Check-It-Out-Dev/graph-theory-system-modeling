-- liquibase formatted sql
-- changeset norbert_marchewka:currency-display-order

ALTER TABLE currency ADD COLUMN IF NOT EXISTS display_order INT NOT NULL DEFAULT 0;

UPDATE currency SET display_order = 0 WHERE iso_code = 'PLN';
UPDATE currency SET display_order = 1 WHERE iso_code = 'EUR';
UPDATE currency SET display_order = 2 WHERE iso_code = 'USD';

-- rollback ALTER TABLE currency DROP COLUMN display_order;
