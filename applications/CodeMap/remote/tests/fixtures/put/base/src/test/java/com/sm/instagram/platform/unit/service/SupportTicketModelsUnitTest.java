package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.ticket.models.*;
import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import jakarta.validation.ValidatorFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

import java.time.LocalDateTime;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for Support Ticket models including entities, enums, and validation.
 * Tests validation, helper methods, and state machine behavior.
 */
@DisplayName("Support Ticket Models Unit Tests")
class SupportTicketModelsUnitTest {

    private Validator validator;

    @BeforeEach
    void setUp() {
        ValidatorFactory factory = Validation.buildDefaultValidatorFactory();
        validator = factory.getValidator();
    }

    // ==================== TicketStatus Enum Tests ====================

    @Nested
    @DisplayName("TicketStatus Enum")
    class TicketStatusEnumTests {

        @Test
        @DisplayName("should have 5 status values")
        void shouldHaveFiveStatusValues() {
            assertThat(TicketStatus.values()).hasSize(5);
        }

        @ParameterizedTest
        @EnumSource(TicketStatus.class)
        @DisplayName("should be able to valueOf all statuses")
        void shouldBeAbleToValueOfAllStatuses(TicketStatus status) {
            assertThat(TicketStatus.valueOf(status.name())).isEqualTo(status);
        }

        @Test
        @DisplayName("should have OPEN status")
        void shouldHaveOpenStatus() {
            assertThat(TicketStatus.OPEN).isNotNull();
            assertThat(TicketStatus.OPEN.name()).isEqualTo("OPEN");
        }

        @Test
        @DisplayName("should have IN_PROGRESS status")
        void shouldHaveInProgressStatus() {
            assertThat(TicketStatus.IN_PROGRESS).isNotNull();
            assertThat(TicketStatus.IN_PROGRESS.name()).isEqualTo("IN_PROGRESS");
        }

        @Test
        @DisplayName("should have WAITING_FOR_CUSTOMER status")
        void shouldHaveWaitingForCustomerStatus() {
            assertThat(TicketStatus.WAITING_FOR_CUSTOMER).isNotNull();
        }

        @Test
        @DisplayName("should have RESOLVED status")
        void shouldHaveResolvedStatus() {
            assertThat(TicketStatus.RESOLVED).isNotNull();
        }

        @Test
        @DisplayName("should have CLOSED status")
        void shouldHaveClosedStatus() {
            assertThat(TicketStatus.CLOSED).isNotNull();
        }

        @Nested
        @DisplayName("Display Names")
        class DisplayNameTests {

            @ParameterizedTest
            @CsvSource({
                    "OPEN, Open",
                    "IN_PROGRESS, In Progress",
                    "WAITING_FOR_CUSTOMER, Awaiting Your Reply",
                    "RESOLVED, Resolved",
                    "CLOSED, Closed"
            })
            @DisplayName("should return correct display name")
            void shouldReturnCorrectDisplayName(TicketStatus status, String expectedDisplayName) {
                assertThat(status.getDisplayName()).isEqualTo(expectedDisplayName);
            }
        }

        @Nested
        @DisplayName("State Transitions")
        class StateTransitionTests {

            @Test
            @DisplayName("OPEN should transition to IN_PROGRESS, RESOLVED, or CLOSED")
            void openShouldTransitionToInProgressResolvedOrClosed() {
                assertThat(TicketStatus.OPEN.canTransitionTo(TicketStatus.IN_PROGRESS)).isTrue();
                assertThat(TicketStatus.OPEN.canTransitionTo(TicketStatus.RESOLVED)).isTrue();
                assertThat(TicketStatus.OPEN.canTransitionTo(TicketStatus.CLOSED)).isTrue();
                assertThat(TicketStatus.OPEN.canTransitionTo(TicketStatus.WAITING_FOR_CUSTOMER)).isFalse();
                assertThat(TicketStatus.OPEN.canTransitionTo(TicketStatus.OPEN)).isFalse();
            }

            @Test
            @DisplayName("IN_PROGRESS should transition to WAITING_FOR_CUSTOMER, RESOLVED, or CLOSED")
            void inProgressShouldTransitionToWaitingResolvedOrClosed() {
                assertThat(TicketStatus.IN_PROGRESS.canTransitionTo(TicketStatus.WAITING_FOR_CUSTOMER)).isTrue();
                assertThat(TicketStatus.IN_PROGRESS.canTransitionTo(TicketStatus.RESOLVED)).isTrue();
                assertThat(TicketStatus.IN_PROGRESS.canTransitionTo(TicketStatus.CLOSED)).isTrue();
                assertThat(TicketStatus.IN_PROGRESS.canTransitionTo(TicketStatus.OPEN)).isFalse();
                assertThat(TicketStatus.IN_PROGRESS.canTransitionTo(TicketStatus.IN_PROGRESS)).isFalse();
            }

            @Test
            @DisplayName("WAITING_FOR_CUSTOMER should transition to IN_PROGRESS, RESOLVED, or CLOSED")
            void waitingForCustomerShouldTransitionToInProgressResolvedOrClosed() {
                assertThat(TicketStatus.WAITING_FOR_CUSTOMER.canTransitionTo(TicketStatus.IN_PROGRESS)).isTrue();
                assertThat(TicketStatus.WAITING_FOR_CUSTOMER.canTransitionTo(TicketStatus.RESOLVED)).isTrue();
                assertThat(TicketStatus.WAITING_FOR_CUSTOMER.canTransitionTo(TicketStatus.CLOSED)).isTrue();
                assertThat(TicketStatus.WAITING_FOR_CUSTOMER.canTransitionTo(TicketStatus.OPEN)).isFalse();
                assertThat(TicketStatus.WAITING_FOR_CUSTOMER.canTransitionTo(TicketStatus.WAITING_FOR_CUSTOMER)).isFalse();
            }

            @Test
            @DisplayName("RESOLVED should transition to IN_PROGRESS, WAITING_FOR_CUSTOMER, or CLOSED")
            void resolvedShouldTransitionToInProgressWaitingOrClosed() {
                assertThat(TicketStatus.RESOLVED.canTransitionTo(TicketStatus.IN_PROGRESS)).isTrue();
                assertThat(TicketStatus.RESOLVED.canTransitionTo(TicketStatus.WAITING_FOR_CUSTOMER)).isTrue();
                assertThat(TicketStatus.RESOLVED.canTransitionTo(TicketStatus.CLOSED)).isTrue();
                assertThat(TicketStatus.RESOLVED.canTransitionTo(TicketStatus.OPEN)).isFalse();
                assertThat(TicketStatus.RESOLVED.canTransitionTo(TicketStatus.RESOLVED)).isFalse();
            }

            @Test
            @DisplayName("CLOSED should not transition to any status")
            void closedShouldNotTransitionToAnyStatus() {
                assertThat(TicketStatus.CLOSED.canTransitionTo(TicketStatus.OPEN)).isFalse();
                assertThat(TicketStatus.CLOSED.canTransitionTo(TicketStatus.IN_PROGRESS)).isFalse();
                assertThat(TicketStatus.CLOSED.canTransitionTo(TicketStatus.WAITING_FOR_CUSTOMER)).isFalse();
                assertThat(TicketStatus.CLOSED.canTransitionTo(TicketStatus.RESOLVED)).isFalse();
                assertThat(TicketStatus.CLOSED.canTransitionTo(TicketStatus.CLOSED)).isFalse();
            }

            @ParameterizedTest
            @EnumSource(TicketStatus.class)
            @DisplayName("no status should transition to itself")
            void noStatusShouldTransitionToItself(TicketStatus status) {
                assertThat(status.canTransitionTo(status)).isFalse();
            }
        }
    }

    // ==================== TicketCategory Enum Tests ====================

    @Nested
    @DisplayName("TicketCategory Enum")
    class TicketCategoryEnumTests {

        @Test
        @DisplayName("should have 9 category values")
        void shouldHaveNineCategoryValues() {
            assertThat(TicketCategory.values()).hasSize(9);
        }

        @ParameterizedTest
        @EnumSource(TicketCategory.class)
        @DisplayName("should be able to valueOf all categories")
        void shouldBeAbleToValueOfAllCategories(TicketCategory category) {
            assertThat(TicketCategory.valueOf(category.name())).isEqualTo(category);
        }

        @ParameterizedTest
        @CsvSource({
                "ACCOUNT_ISSUE, Account Issue",
                "BILLING_PAYMENT, Billing & Payment",
                "TECHNICAL_PROBLEM, Technical Problem",
                "FEATURE_REQUEST, Feature Request",
                "PARTNERSHIP_ISSUE, Partnership Issue",
                "CONTENT_MODERATION, Content Moderation",
                "GENERAL_INQUIRY, General Inquiry",
                "EARLY_ACCESS_INTEREST, Early Access Interest",
                "OTHER, Other"
        })
        @DisplayName("should return correct display name for category")
        void shouldReturnCorrectDisplayNameForCategory(TicketCategory category, String expectedDisplayName) {
            assertThat(category.getDisplayName()).isEqualTo(expectedDisplayName);
        }

        @Test
        @DisplayName("should have all expected categories")
        void shouldHaveAllExpectedCategories() {
            assertThat(TicketCategory.ACCOUNT_ISSUE).isNotNull();
            assertThat(TicketCategory.BILLING_PAYMENT).isNotNull();
            assertThat(TicketCategory.TECHNICAL_PROBLEM).isNotNull();
            assertThat(TicketCategory.FEATURE_REQUEST).isNotNull();
            assertThat(TicketCategory.PARTNERSHIP_ISSUE).isNotNull();
            assertThat(TicketCategory.CONTENT_MODERATION).isNotNull();
            assertThat(TicketCategory.GENERAL_INQUIRY).isNotNull();
            assertThat(TicketCategory.EARLY_ACCESS_INTEREST).isNotNull();
            assertThat(TicketCategory.OTHER).isNotNull();
        }
    }

    // ==================== SupportTicket Entity Tests ====================

    @Nested
    @DisplayName("SupportTicket Entity")
    class SupportTicketEntityTests {

        private SupportTicket ticket;

        @BeforeEach
        void setUp() {
            ticket = new SupportTicket();
            ticket.setContactEmail("user@example.com");
            ticket.setSubject("Need help with account");
            ticket.setDescription("I cannot log in to my account.");
            ticket.setStatus(TicketStatus.OPEN);
            ticket.setCategory(TicketCategory.ACCOUNT_ISSUE);
            ticket.setTicketReference("TKT-12345");
        }

        @Nested
        @DisplayName("Validation")
        class ValidationTests {

            @Test
            @DisplayName("should pass validation for valid ticket")
            void shouldPassValidationForValidTicket() {
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isEmpty();
            }

            @ParameterizedTest
            @NullAndEmptySource
            @ValueSource(strings = {"   ", "\t"})
            @DisplayName("should fail validation for blank contact email")
            void shouldFailValidationForBlankContactEmail(String email) {
                ticket.setContactEmail(email);
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
                assertThat(violations.stream()
                        .anyMatch(v -> v.getPropertyPath().toString().equals("contactEmail"))).isTrue();
            }

            @ParameterizedTest
            @ValueSource(strings = {"invalid", "invalid@", "@example.com", "user@@example.com"})
            @DisplayName("should fail validation for invalid email format")
            void shouldFailValidationForInvalidEmailFormat(String email) {
                ticket.setContactEmail(email);
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
            }

            @Test
            @DisplayName("should fail validation for email exceeding 255 characters")
            void shouldFailValidationForLongEmail() {
                ticket.setContactEmail("a".repeat(250) + "@example.com");
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
            }

            @ParameterizedTest
            @NullAndEmptySource
            @ValueSource(strings = {"   ", "\t"})
            @DisplayName("should fail validation for blank subject")
            void shouldFailValidationForBlankSubject(String subject) {
                ticket.setSubject(subject);
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
                assertThat(violations.stream()
                        .anyMatch(v -> v.getPropertyPath().toString().equals("subject"))).isTrue();
            }

            @Test
            @DisplayName("should fail validation for subject exceeding 255 characters")
            void shouldFailValidationForLongSubject() {
                ticket.setSubject("A".repeat(256));
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
            }

            @ParameterizedTest
            @NullAndEmptySource
            @ValueSource(strings = {"   ", "\t"})
            @DisplayName("should fail validation for blank description")
            void shouldFailValidationForBlankDescription(String description) {
                ticket.setDescription(description);
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
                assertThat(violations.stream()
                        .anyMatch(v -> v.getPropertyPath().toString().equals("description"))).isTrue();
            }

            @Test
            @DisplayName("should fail validation for description exceeding 5000 characters")
            void shouldFailValidationForLongDescription() {
                ticket.setDescription("A".repeat(5001));
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
                assertThat(violations.stream()
                        .anyMatch(v -> v.getMessage().contains("5000"))).isTrue();
            }

            @Test
            @DisplayName("should pass validation for description with exactly 5000 characters")
            void shouldPassValidationForMaxLengthDescription() {
                ticket.setDescription("A".repeat(5000));
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isEmpty();
            }

            @Test
            @DisplayName("should fail validation for technical description exceeding 100000 characters")
            void shouldFailValidationForLongTechnicalDescription() {
                ticket.setTechnicalDescription("A".repeat(100001));
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
            }

            @Test
            @DisplayName("should pass validation for technical description with 100000 characters")
            void shouldPassValidationForMaxLengthTechnicalDescription() {
                ticket.setTechnicalDescription("A".repeat(100000));
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isEmpty();
            }

            @Test
            @DisplayName("should fail validation for null status")
            void shouldFailValidationForNullStatus() {
                ticket.setStatus(null);
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
                assertThat(violations.stream()
                        .anyMatch(v -> v.getPropertyPath().toString().equals("status"))).isTrue();
            }

            @Test
            @DisplayName("should fail validation for null category")
            void shouldFailValidationForNullCategory() {
                ticket.setCategory(null);
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
                assertThat(violations.stream()
                        .anyMatch(v -> v.getPropertyPath().toString().equals("category"))).isTrue();
            }

            @ParameterizedTest
            @NullAndEmptySource
            @ValueSource(strings = {"   ", "\t"})
            @DisplayName("should fail validation for blank ticket reference")
            void shouldFailValidationForBlankTicketReference(String reference) {
                ticket.setTicketReference(reference);
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
                assertThat(violations.stream()
                        .anyMatch(v -> v.getPropertyPath().toString().equals("ticketReference"))).isTrue();
            }

            @Test
            @DisplayName("should fail validation for ticket reference exceeding 32 characters")
            void shouldFailValidationForLongTicketReference() {
                // Widened from 20 for the pentest 3.3 format CIO-yyyyMMdd-XXXXXXXX (21 chars).
                ticket.setTicketReference("A".repeat(33));
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
            }

            @Test
            @DisplayName("should fail validation for updater ID exceeding 255 characters")
            void shouldFailValidationForLongUpdaterId() {
                ticket.setUpdaterId("A".repeat(256));
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isNotEmpty();
            }

            @Test
            @DisplayName("should pass validation with null optional fields")
            void shouldPassValidationWithNullOptionalFields() {
                ticket.setTechnicalDescription(null);
                ticket.setIpAddress(null);
                ticket.setAdminAssignee(null);
                ticket.setUpdaterId(null);
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations).isEmpty();
            }
        }

        @Nested
        @DisplayName("Default Values")
        class DefaultValueTests {

            @Test
            @DisplayName("should have default status as OPEN")
            void shouldHaveDefaultStatusOpen() {
                SupportTicket newTicket = new SupportTicket();
                assertThat(newTicket.getStatus()).isEqualTo(TicketStatus.OPEN);
            }

            @Test
            @DisplayName("should have empty responses list by default")
            void shouldHaveEmptyResponsesListByDefault() {
                SupportTicket newTicket = new SupportTicket();
                assertThat(newTicket.getResponses()).isNotNull().isEmpty();
            }

            @Test
            @DisplayName("should have empty attachments list by default")
            void shouldHaveEmptyAttachmentsListByDefault() {
                SupportTicket newTicket = new SupportTicket();
                assertThat(newTicket.getAttachments()).isNotNull().isEmpty();
            }
        }

        @Nested
        @DisplayName("Helper Methods")
        class HelperMethodTests {

            @Test
            @DisplayName("addResponse should add response and set ticket reference")
            void addResponseShouldAddResponseAndSetTicketReference() {
                TicketResponse response = new TicketResponse();
                response.setContent("Thank you for contacting us.");

                ticket.addResponse(response);

                assertThat(ticket.getResponses()).contains(response);
                assertThat(response.getTicket()).isEqualTo(ticket);
            }

            @Test
            @DisplayName("addResponse should handle multiple responses")
            void addResponseShouldHandleMultipleResponses() {
                TicketResponse response1 = new TicketResponse();
                response1.setContent("First response");

                TicketResponse response2 = new TicketResponse();
                response2.setContent("Second response");

                ticket.addResponse(response1);
                ticket.addResponse(response2);

                assertThat(ticket.getResponses()).hasSize(2);
                assertThat(ticket.getResponses()).containsExactly(response1, response2);
            }

            @Test
            @DisplayName("addAttachment should add attachment and set ticket reference")
            void addAttachmentShouldAddAttachmentAndSetTicketReference() {
                TicketAttachment attachment = new TicketAttachment();
                attachment.setFileName("screenshot.png");
                attachment.setContentType("image/png");
                attachment.setFileUrl("https://storage.example.com/screenshot.png");
                attachment.setFileSize(1024L);

                ticket.addAttachment(attachment);

                assertThat(ticket.getAttachments()).contains(attachment);
                assertThat(attachment.getTicket()).isEqualTo(ticket);
            }

            @Test
            @DisplayName("addAttachment should handle multiple attachments")
            void addAttachmentShouldHandleMultipleAttachments() {
                TicketAttachment attachment1 = new TicketAttachment();
                attachment1.setFileName("file1.png");
                attachment1.setContentType("image/png");
                attachment1.setFileUrl("https://storage.example.com/file1.png");
                attachment1.setFileSize(1024L);

                TicketAttachment attachment2 = new TicketAttachment();
                attachment2.setFileName("file2.pdf");
                attachment2.setContentType("application/pdf");
                attachment2.setFileUrl("https://storage.example.com/file2.pdf");
                attachment2.setFileSize(2048L);

                ticket.addAttachment(attachment1);
                ticket.addAttachment(attachment2);

                assertThat(ticket.getAttachments()).hasSize(2);
            }

            @Test
            @DisplayName("isResolved should return true for RESOLVED status")
            void isResolvedShouldReturnTrueForResolvedStatus() {
                ticket.setStatus(TicketStatus.RESOLVED);
                assertThat(ticket.isResolved()).isTrue();
            }

            @Test
            @DisplayName("isResolved should return true for CLOSED status")
            void isResolvedShouldReturnTrueForClosedStatus() {
                ticket.setStatus(TicketStatus.CLOSED);
                assertThat(ticket.isResolved()).isTrue();
            }

            @Test
            @DisplayName("isResolved should return false for OPEN status")
            void isResolvedShouldReturnFalseForOpenStatus() {
                ticket.setStatus(TicketStatus.OPEN);
                assertThat(ticket.isResolved()).isFalse();
            }

            @Test
            @DisplayName("isResolved should return false for IN_PROGRESS status")
            void isResolvedShouldReturnFalseForInProgressStatus() {
                ticket.setStatus(TicketStatus.IN_PROGRESS);
                assertThat(ticket.isResolved()).isFalse();
            }

            @Test
            @DisplayName("isResolved should return false for WAITING_FOR_CUSTOMER status")
            void isResolvedShouldReturnFalseForWaitingForCustomerStatus() {
                ticket.setStatus(TicketStatus.WAITING_FOR_CUSTOMER);
                assertThat(ticket.isResolved()).isFalse();
            }
        }

        @Nested
        @DisplayName("Setters and Getters")
        class SettersAndGettersTests {

            @Test
            @DisplayName("should set and get all fields")
            void shouldSetAndGetAllFields() {
                LocalDateTime now = LocalDateTime.now();

                ticket.setId(1L);
                ticket.setContactEmail("test@example.com");
                ticket.setSubject("Test Subject");
                ticket.setDescription("Test Description");
                ticket.setTechnicalDescription("Technical details");
                ticket.setStatus(TicketStatus.IN_PROGRESS);
                ticket.setCategory(TicketCategory.TECHNICAL_PROBLEM);
                ticket.setIpAddress("192.168.1.1");
                ticket.setTicketReference("TKT-99999");
                ticket.setAdminAssignee("admin@example.com");
                ticket.setResolvedTime(now);
                ticket.setUpdaterId("updater-123");

                assertThat(ticket.getId()).isEqualTo(1L);
                assertThat(ticket.getContactEmail()).isEqualTo("test@example.com");
                assertThat(ticket.getSubject()).isEqualTo("Test Subject");
                assertThat(ticket.getDescription()).isEqualTo("Test Description");
                assertThat(ticket.getTechnicalDescription()).isEqualTo("Technical details");
                assertThat(ticket.getStatus()).isEqualTo(TicketStatus.IN_PROGRESS);
                assertThat(ticket.getCategory()).isEqualTo(TicketCategory.TECHNICAL_PROBLEM);
                assertThat(ticket.getIpAddress()).isEqualTo("192.168.1.1");
                assertThat(ticket.getTicketReference()).isEqualTo("TKT-99999");
                assertThat(ticket.getAdminAssignee()).isEqualTo("admin@example.com");
                assertThat(ticket.getResolvedTime()).isEqualTo(now);
                assertThat(ticket.getUpdaterId()).isEqualTo("updater-123");
            }
        }

        @Nested
        @DisplayName("Email Validation Scenarios")
        class EmailValidationScenarios {

            @ParameterizedTest
            @ValueSource(strings = {
                    "user@example.com",
                    "user.name@example.com",
                    "user+tag@example.com",
                    "user@subdomain.example.com",
                    "user@example.co.uk"
            })
            @DisplayName("should accept valid email formats")
            void shouldAcceptValidEmailFormats(String email) {
                ticket.setContactEmail(email);
                Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
                assertThat(violations.stream()
                        .noneMatch(v -> v.getPropertyPath().toString().equals("contactEmail"))).isTrue();
            }
        }
    }

    // ==================== TicketAttachment Entity Tests ====================

    @Nested
    @DisplayName("TicketAttachment Entity")
    class TicketAttachmentEntityTests {

        private TicketAttachment attachment;

        @BeforeEach
        void setUp() {
            attachment = new TicketAttachment();
            attachment.setFileName("document.pdf");
            attachment.setContentType("application/pdf");
            attachment.setFileUrl("https://storage.example.com/files/document.pdf");
            attachment.setFileSize(10240L);
        }

        @Nested
        @DisplayName("Validation")
        class ValidationTests {

            @Test
            @DisplayName("should pass validation for valid attachment")
            void shouldPassValidationForValidAttachment() {
                Set<ConstraintViolation<TicketAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isEmpty();
            }

            @ParameterizedTest
            @NullAndEmptySource
            @ValueSource(strings = {"   ", "\t"})
            @DisplayName("should fail validation for blank file name")
            void shouldFailValidationForBlankFileName(String fileName) {
                attachment.setFileName(fileName);
                Set<ConstraintViolation<TicketAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
                assertThat(violations.stream()
                        .anyMatch(v -> v.getPropertyPath().toString().equals("fileName"))).isTrue();
            }

            @Test
            @DisplayName("should fail validation for file name exceeding 255 characters")
            void shouldFailValidationForLongFileName() {
                attachment.setFileName("A".repeat(256));
                Set<ConstraintViolation<TicketAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
            }

            @ParameterizedTest
            @NullAndEmptySource
            @ValueSource(strings = {"   ", "\t"})
            @DisplayName("should fail validation for blank content type")
            void shouldFailValidationForBlankContentType(String contentType) {
                attachment.setContentType(contentType);
                Set<ConstraintViolation<TicketAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
                assertThat(violations.stream()
                        .anyMatch(v -> v.getPropertyPath().toString().equals("contentType"))).isTrue();
            }

            @Test
            @DisplayName("should fail validation for content type exceeding 100 characters")
            void shouldFailValidationForLongContentType() {
                attachment.setContentType("A".repeat(101));
                Set<ConstraintViolation<TicketAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
            }

            @ParameterizedTest
            @NullAndEmptySource
            @ValueSource(strings = {"   ", "\t"})
            @DisplayName("should fail validation for blank file URL")
            void shouldFailValidationForBlankFileUrl(String fileUrl) {
                attachment.setFileUrl(fileUrl);
                Set<ConstraintViolation<TicketAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
                assertThat(violations.stream()
                        .anyMatch(v -> v.getPropertyPath().toString().equals("fileUrl"))).isTrue();
            }

            @Test
            @DisplayName("should fail validation for file URL exceeding 2048 characters")
            void shouldFailValidationForLongFileUrl() {
                attachment.setFileUrl("https://example.com/" + "A".repeat(2030));
                Set<ConstraintViolation<TicketAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
            }

            @Test
            @DisplayName("should fail validation for zero file size")
            void shouldFailValidationForZeroFileSize() {
                attachment.setFileSize(0L);
                Set<ConstraintViolation<TicketAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
                assertThat(violations.stream()
                        .anyMatch(v -> v.getPropertyPath().toString().equals("fileSize"))).isTrue();
            }

            @Test
            @DisplayName("should fail validation for negative file size")
            void shouldFailValidationForNegativeFileSize() {
                attachment.setFileSize(-100L);
                Set<ConstraintViolation<TicketAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
            }

            @Test
            @DisplayName("should pass validation for positive file size")
            void shouldPassValidationForPositiveFileSize() {
                attachment.setFileSize(1L);
                Set<ConstraintViolation<TicketAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isEmpty();
            }
        }

        @Nested
        @DisplayName("Setters and Getters")
        class SettersAndGettersTests {

            @Test
            @DisplayName("should set and get all fields")
            void shouldSetAndGetAllFields() {
                SupportTicket ticket = new SupportTicket();
                ticket.setId(1L);

                attachment.setId(10L);
                attachment.setTicket(ticket);
                attachment.setFileName("image.png");
                attachment.setContentType("image/png");
                attachment.setFileUrl("https://storage.example.com/image.png");
                attachment.setFileSize(5000L);

                assertThat(attachment.getId()).isEqualTo(10L);
                assertThat(attachment.getTicket()).isEqualTo(ticket);
                assertThat(attachment.getFileName()).isEqualTo("image.png");
                assertThat(attachment.getContentType()).isEqualTo("image/png");
                assertThat(attachment.getFileUrl()).isEqualTo("https://storage.example.com/image.png");
                assertThat(attachment.getFileSize()).isEqualTo(5000L);
            }
        }

        @Nested
        @DisplayName("Content Type Scenarios")
        class ContentTypeScenarios {

            @ParameterizedTest
            @CsvSource({
                    "screenshot.png, image/png",
                    "document.pdf, application/pdf",
                    "log.txt, text/plain",
                    "data.json, application/json",
                    "archive.zip, application/zip"
            })
            @DisplayName("should accept various content types")
            void shouldAcceptVariousContentTypes(String fileName, String contentType) {
                attachment.setFileName(fileName);
                attachment.setContentType(contentType);
                Set<ConstraintViolation<TicketAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isEmpty();
            }
        }
    }

    // ==================== TicketResponse Entity Tests ====================

    @Nested
    @DisplayName("TicketResponse Entity")
    class TicketResponseEntityTests {

        private TicketResponse response;

        @BeforeEach
        void setUp() {
            response = new TicketResponse();
            response.setContent("Thank you for contacting support. We are looking into your issue.");
        }

        @Nested
        @DisplayName("Validation")
        class ValidationTests {

            @Test
            @DisplayName("should pass validation for valid response")
            void shouldPassValidationForValidResponse() {
                Set<ConstraintViolation<TicketResponse>> violations = validator.validate(response);
                assertThat(violations).isEmpty();
            }

            @ParameterizedTest
            @NullAndEmptySource
            @ValueSource(strings = {"   ", "\t"})
            @DisplayName("should fail validation for blank content")
            void shouldFailValidationForBlankContent(String content) {
                response.setContent(content);
                Set<ConstraintViolation<TicketResponse>> violations = validator.validate(response);
                assertThat(violations).isNotEmpty();
                assertThat(violations.stream()
                        .anyMatch(v -> v.getPropertyPath().toString().equals("content"))).isTrue();
            }
        }

        @Nested
        @DisplayName("Default Values")
        class DefaultValueTests {

            @Test
            @DisplayName("should have empty attachments list by default")
            void shouldHaveEmptyAttachmentsListByDefault() {
                TicketResponse newResponse = new TicketResponse();
                assertThat(newResponse.getAttachments()).isNotNull().isEmpty();
            }
        }

        @Nested
        @DisplayName("Helper Methods")
        class HelperMethodTests {

            @Test
            @DisplayName("addAttachment should add attachment and set response reference")
            void addAttachmentShouldAddAttachmentAndSetResponseReference() {
                ResponseAttachment attachment = new ResponseAttachment();
                attachment.setFileName("image.png");
                attachment.setContentType("image/png");
                attachment.setFileUrl("https://storage.example.com/image.png");
                attachment.setFileSize(1024L);

                response.addAttachment(attachment);

                assertThat(response.getAttachments()).contains(attachment);
                assertThat(attachment.getResponse()).isEqualTo(response);
            }

            @Test
            @DisplayName("addAttachment should handle multiple attachments")
            void addAttachmentShouldHandleMultipleAttachments() {
                ResponseAttachment attachment1 = new ResponseAttachment();
                attachment1.setFileName("file1.png");
                attachment1.setContentType("image/png");
                attachment1.setFileUrl("https://storage.example.com/file1.png");
                attachment1.setFileSize(1024L);

                ResponseAttachment attachment2 = new ResponseAttachment();
                attachment2.setFileName("file2.pdf");
                attachment2.setContentType("application/pdf");
                attachment2.setFileUrl("https://storage.example.com/file2.pdf");
                attachment2.setFileSize(2048L);

                response.addAttachment(attachment1);
                response.addAttachment(attachment2);

                assertThat(response.getAttachments()).hasSize(2);
            }
        }

        @Nested
        @DisplayName("Setters and Getters")
        class SettersAndGettersTests {

            @Test
            @DisplayName("should set and get all fields")
            void shouldSetAndGetAllFields() {
                SupportTicket ticket = new SupportTicket();
                ticket.setId(1L);

                response.setId(10L);
                response.setTicket(ticket);
                response.setContent("Response content");
                response.setFromAdmin(true);
                response.setAdminName("John Doe");
                response.setEmailSent(true);

                assertThat(response.getId()).isEqualTo(10L);
                assertThat(response.getTicket()).isEqualTo(ticket);
                assertThat(response.getContent()).isEqualTo("Response content");
                assertThat(response.isFromAdmin()).isTrue();
                assertThat(response.getAdminName()).isEqualTo("John Doe");
                assertThat(response.isEmailSent()).isTrue();
            }

            @Test
            @DisplayName("should handle customer response (not from admin)")
            void shouldHandleCustomerResponse() {
                response.setFromAdmin(false);
                response.setAdminName(null);

                assertThat(response.isFromAdmin()).isFalse();
                assertThat(response.getAdminName()).isNull();
            }
        }
    }

    // ==================== ResponseAttachment Entity Tests ====================

    @Nested
    @DisplayName("ResponseAttachment Entity")
    class ResponseAttachmentEntityTests {

        private ResponseAttachment attachment;

        @BeforeEach
        void setUp() {
            attachment = new ResponseAttachment();
            attachment.setFileName("screenshot.png");
            attachment.setContentType("image/png");
            attachment.setFileUrl("https://storage.example.com/responses/screenshot.png");
            attachment.setFileSize(2048L);
        }

        @Nested
        @DisplayName("Validation")
        class ValidationTests {

            @Test
            @DisplayName("should pass validation for valid attachment")
            void shouldPassValidationForValidAttachment() {
                Set<ConstraintViolation<ResponseAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isEmpty();
            }

            @ParameterizedTest
            @NullAndEmptySource
            @ValueSource(strings = {"   ", "\t"})
            @DisplayName("should fail validation for blank file name")
            void shouldFailValidationForBlankFileName(String fileName) {
                attachment.setFileName(fileName);
                Set<ConstraintViolation<ResponseAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
            }

            @Test
            @DisplayName("should fail validation for file name exceeding 255 characters")
            void shouldFailValidationForLongFileName() {
                attachment.setFileName("A".repeat(256));
                Set<ConstraintViolation<ResponseAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
            }

            @ParameterizedTest
            @NullAndEmptySource
            @ValueSource(strings = {"   ", "\t"})
            @DisplayName("should fail validation for blank content type")
            void shouldFailValidationForBlankContentType(String contentType) {
                attachment.setContentType(contentType);
                Set<ConstraintViolation<ResponseAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
            }

            @Test
            @DisplayName("should fail validation for content type exceeding 100 characters")
            void shouldFailValidationForLongContentType() {
                attachment.setContentType("A".repeat(101));
                Set<ConstraintViolation<ResponseAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
            }

            @ParameterizedTest
            @NullAndEmptySource
            @ValueSource(strings = {"   ", "\t"})
            @DisplayName("should fail validation for blank file URL")
            void shouldFailValidationForBlankFileUrl(String fileUrl) {
                attachment.setFileUrl(fileUrl);
                Set<ConstraintViolation<ResponseAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
            }

            @Test
            @DisplayName("should fail validation for file URL exceeding 2048 characters")
            void shouldFailValidationForLongFileUrl() {
                attachment.setFileUrl("https://example.com/" + "A".repeat(2030));
                Set<ConstraintViolation<ResponseAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
            }

            @Test
            @DisplayName("should fail validation for zero file size")
            void shouldFailValidationForZeroFileSize() {
                attachment.setFileSize(0L);
                Set<ConstraintViolation<ResponseAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
            }

            @Test
            @DisplayName("should fail validation for negative file size")
            void shouldFailValidationForNegativeFileSize() {
                attachment.setFileSize(-100L);
                Set<ConstraintViolation<ResponseAttachment>> violations = validator.validate(attachment);
                assertThat(violations).isNotEmpty();
            }
        }

        @Nested
        @DisplayName("Setters and Getters")
        class SettersAndGettersTests {

            @Test
            @DisplayName("should set and get all fields")
            void shouldSetAndGetAllFields() {
                TicketResponse response = new TicketResponse();
                response.setId(1L);

                attachment.setId(10L);
                attachment.setResponse(response);
                attachment.setFileName("document.pdf");
                attachment.setContentType("application/pdf");
                attachment.setFileUrl("https://storage.example.com/document.pdf");
                attachment.setFileSize(50000L);

                assertThat(attachment.getId()).isEqualTo(10L);
                assertThat(attachment.getResponse()).isEqualTo(response);
                assertThat(attachment.getFileName()).isEqualTo("document.pdf");
                assertThat(attachment.getContentType()).isEqualTo("application/pdf");
                assertThat(attachment.getFileUrl()).isEqualTo("https://storage.example.com/document.pdf");
                assertThat(attachment.getFileSize()).isEqualTo(50000L);
            }
        }
    }

    // ==================== Edge Cases ====================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("should handle ticket with maximum length fields")
        void shouldHandleTicketWithMaximumLengthFields() {
            SupportTicket ticket = new SupportTicket();
            ticket.setContactEmail("maxlengthuser@example.com"); // Valid email for max-length test
            ticket.setSubject("A".repeat(255));
            ticket.setDescription("A".repeat(5000));
            ticket.setTechnicalDescription("A".repeat(100000));
            ticket.setTicketReference("A".repeat(20));
            ticket.setStatus(TicketStatus.OPEN);
            ticket.setCategory(TicketCategory.OTHER);

            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }

        @Test
        @DisplayName("should handle Unicode characters in description")
        void shouldHandleUnicodeCharactersInDescription() {
            SupportTicket ticket = new SupportTicket();
            ticket.setContactEmail("user@example.com");
            ticket.setSubject("Problem z kontem");
            ticket.setDescription("Nie mogę zalogować się do mojego konta. Proszę o pomoc w rozwiązaniu problemu.");
            ticket.setStatus(TicketStatus.OPEN);
            ticket.setCategory(TicketCategory.ACCOUNT_ISSUE);
            ticket.setTicketReference("TKT-12345");

            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }

        @Test
        @DisplayName("should handle ticket workflow from OPEN to CLOSED")
        void shouldHandleTicketWorkflowFromOpenToClosed() {
            SupportTicket ticket = new SupportTicket();
            ticket.setContactEmail("user@example.com");
            ticket.setSubject("Issue");
            ticket.setDescription("Description");
            ticket.setStatus(TicketStatus.OPEN);
            ticket.setCategory(TicketCategory.TECHNICAL_PROBLEM);
            ticket.setTicketReference("TKT-12345");

            // Simulate workflow
            assertThat(ticket.getStatus().canTransitionTo(TicketStatus.IN_PROGRESS)).isTrue();
            ticket.setStatus(TicketStatus.IN_PROGRESS);

            assertThat(ticket.getStatus().canTransitionTo(TicketStatus.WAITING_FOR_CUSTOMER)).isTrue();
            ticket.setStatus(TicketStatus.WAITING_FOR_CUSTOMER);

            assertThat(ticket.getStatus().canTransitionTo(TicketStatus.IN_PROGRESS)).isTrue();
            ticket.setStatus(TicketStatus.IN_PROGRESS);

            assertThat(ticket.getStatus().canTransitionTo(TicketStatus.RESOLVED)).isTrue();
            ticket.setStatus(TicketStatus.RESOLVED);
            assertThat(ticket.isResolved()).isTrue();

            assertThat(ticket.getStatus().canTransitionTo(TicketStatus.CLOSED)).isTrue();
            ticket.setStatus(TicketStatus.CLOSED);
            assertThat(ticket.isResolved()).isTrue();

            // Cannot transition from CLOSED
            assertThat(ticket.getStatus().canTransitionTo(TicketStatus.OPEN)).isFalse();
        }

        @Test
        @DisplayName("should handle very long response content")
        void shouldHandleVeryLongResponseContent() {
            TicketResponse response = new TicketResponse();
            response.setContent("A".repeat(100000));

            Set<ConstraintViolation<TicketResponse>> violations = validator.validate(response);
            assertThat(violations).isEmpty();
        }

        @Test
        @DisplayName("should handle special characters in file names")
        void shouldHandleSpecialCharactersInFileNames() {
            TicketAttachment attachment = new TicketAttachment();
            attachment.setFileName("error_log (1) - copy.txt");
            attachment.setContentType("text/plain");
            attachment.setFileUrl("https://storage.example.com/file.txt");
            attachment.setFileSize(100L);

            Set<ConstraintViolation<TicketAttachment>> violations = validator.validate(attachment);
            assertThat(violations).isEmpty();
        }
    }
}
