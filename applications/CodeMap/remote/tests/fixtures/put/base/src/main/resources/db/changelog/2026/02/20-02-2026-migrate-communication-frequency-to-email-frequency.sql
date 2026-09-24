--liquibase formatted sql

--changeset migration:20-02-2026-migrate-communication-frequency-to-email-frequency
--comment: Migrate communication_frequency from old CommunicationFrequency enum (DAILY, WEEKLY, MONTHLY, NEVER) to new EmailFrequency enum (IMMEDIATE, HOURLY_DIGEST, DAILY_DIGEST, WEEKLY_DIGEST)

-- Step 1: Drop the old CHECK constraint
ALTER TABLE user_preferences DROP CONSTRAINT IF EXISTS user_preferences_communication_frequency_check;

-- Step 2: Migrate existing data from old values to new values
UPDATE user_preferences SET communication_frequency = 'DAILY_DIGEST' WHERE communication_frequency = 'DAILY';
UPDATE user_preferences SET communication_frequency = 'WEEKLY_DIGEST' WHERE communication_frequency = 'WEEKLY';
UPDATE user_preferences SET communication_frequency = 'WEEKLY_DIGEST' WHERE communication_frequency = 'MONTHLY';
UPDATE user_preferences SET communication_frequency = 'WEEKLY_DIGEST' WHERE communication_frequency = 'NEVER';

-- Step 3: Update the default value
ALTER TABLE user_preferences ALTER COLUMN communication_frequency SET DEFAULT 'WEEKLY_DIGEST';

-- Step 4: Add new CHECK constraint with EmailFrequency values
ALTER TABLE user_preferences ADD CONSTRAINT user_preferences_communication_frequency_check
    CHECK (communication_frequency IN ('IMMEDIATE', 'HOURLY_DIGEST', 'DAILY_DIGEST', 'WEEKLY_DIGEST'));
