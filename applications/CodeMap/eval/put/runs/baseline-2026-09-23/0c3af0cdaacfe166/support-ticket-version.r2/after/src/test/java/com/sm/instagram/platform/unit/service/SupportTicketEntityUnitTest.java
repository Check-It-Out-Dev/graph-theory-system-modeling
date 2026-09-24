package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.support.ticket.models.*;
import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validation;
import jakarta.validation.Validator;
import jakarta.validation.ValidatorFactory;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Set;

import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive unit tests for SupportTicket entity.
 * Tests validation constraints, helper methods, default values, and bidirectional relationships.
 */
@DisplayName("SupportTicket Entity Unit Tests")
class SupportTicketEntityUnitTest {

    private static Validator validator;
    private SupportTicket ticket;

    @BeforeAll
    static void setUpValidator() {
        ValidatorFactory factory = Validation.buildDefaultValidatorFactory();
        validator = factory.getValidator();
    }

    @BeforeEach
    void setUp() {
        ticket = createValidTicket();
    }

    private SupportTicket createValidTicket() {
        SupportTicket t = new SupportTicket();
        t.setContactEmail("user@example.com");
        t.setSubject("Need help with account");
        t.setDescription("I cannot log in to my account.");
        t.setStatus(TicketStatus.OPEN);
        t.setCategory(TicketCategory.ACCOUNT_ISSUE);
        t.setTicketReference("TKT-12345");
        return t;
    }

    // ==================== contactEmail Validation Tests ====================

    @Nested
    @DisplayName("contactEmail Validation")
    class ContactEmailValidationTests {

        @Test
        @DisplayName("should pass validation for valid email")
        void shouldPassValidationForValidEmail() {
            ticket.setContactEmail("valid@example.com");
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }

        @ParameterizedTest
        @NullAndEmptySource
        @ValueSource(strings = {"   ", "\t", "\n"})
        @DisplayName("should fail validation for blank contactEmail - @NotBlank")
        void shouldFailValidationForBlankContactEmail(String email) {
            ticket.setContactEmail(email);
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("contactEmail"))).isTrue();
        }

        @ParameterizedTest
        @ValueSource(strings = {"invalid", "invalid@", "@example.com", "user@@example.com", "user@.com"})
        @DisplayName("should fail validation for invalid email format - @Email")
        void shouldFailValidationForInvalidEmailFormat(String email) {
            ticket.setContactEmail(email);
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("contactEmail"))).isTrue();
        }

        @Test
        @DisplayName("should fail validation for email exceeding 255 characters - @Size(max=255)")
        void shouldFailValidationForEmailExceeding255Characters() {
            String longEmail = "a".repeat(250) + "@example.com";
            ticket.setContactEmail(longEmail);
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("contactEmail")
                            && v.getMessage().contains("255"))).isTrue();
        }

        @Test
        @DisplayName("should pass validation for email with exactly 255 characters")
        void shouldPassValidationForEmailWithExactly255Characters() {
            // Create email that is exactly 255 characters
            String localPart = "a".repeat(243); // 243 + @example.com (12) = 255
            String email = localPart + "@example.com";
            ticket.setContactEmail(email);
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            // Email format validation may fail but size should be valid
            boolean hasSizeViolation = violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("contactEmail")
                            && v.getMessage().contains("255"));
            assertThat(hasSizeViolation).isFalse();
        }
    }

    // ==================== subject Validation Tests ====================

    @Nested
    @DisplayName("subject Validation")
    class SubjectValidationTests {

        @Test
        @DisplayName("should pass validation for valid subject")
        void shouldPassValidationForValidSubject() {
            ticket.setSubject("Valid subject line");
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }

        @ParameterizedTest
        @NullAndEmptySource
        @ValueSource(strings = {"   ", "\t", "\n"})
        @DisplayName("should fail validation for blank subject - @NotBlank")
        void shouldFailValidationForBlankSubject(String subject) {
            ticket.setSubject(subject);
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("subject"))).isTrue();
        }

        @Test
        @DisplayName("should fail validation for subject exceeding 255 characters - @Size(max=255)")
        void shouldFailValidationForSubjectExceeding255Characters() {
            ticket.setSubject("A".repeat(256));
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("subject"))).isTrue();
        }

        @Test
        @DisplayName("should pass validation for subject with exactly 255 characters")
        void shouldPassValidationForSubjectWithExactly255Characters() {
            ticket.setSubject("A".repeat(255));
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }
    }

    // ==================== description Validation Tests ====================

    @Nested
    @DisplayName("description Validation")
    class DescriptionValidationTests {

        @Test
        @DisplayName("should pass validation for valid description")
        void shouldPassValidationForValidDescription() {
            ticket.setDescription("This is a valid description of the issue.");
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }

        @ParameterizedTest
        @NullAndEmptySource
        @ValueSource(strings = {"   ", "\t", "\n"})
        @DisplayName("should fail validation for blank description - @NotBlank")
        void shouldFailValidationForBlankDescription(String description) {
            ticket.setDescription(description);
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("description"))).isTrue();
        }

        @Test
        @DisplayName("should fail validation for description exceeding 5000 characters - @Size(max=5000)")
        void shouldFailValidationForDescriptionExceeding5000Characters() {
            ticket.setDescription("A".repeat(5001));
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("description")
                            && v.getMessage().contains("5000"))).isTrue();
        }

        @Test
        @DisplayName("should pass validation for description with exactly 5000 characters")
        void shouldPassValidationForDescriptionWithExactly5000Characters() {
            ticket.setDescription("A".repeat(5000));
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }
    }

    // ==================== status Validation Tests ====================

    @Nested
    @DisplayName("status Validation")
    class StatusValidationTests {

        @Test
        @DisplayName("should fail validation for null status - @NotNull")
        void shouldFailValidationForNullStatus() {
            ticket.setStatus(null);
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("status"))).isTrue();
        }

        @ParameterizedTest
        @EnumSource(TicketStatus.class)
        @DisplayName("should pass validation for all valid status values")
        void shouldPassValidationForAllValidStatusValues(TicketStatus status) {
            ticket.setStatus(status);
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations.stream()
                    .noneMatch(v -> v.getPropertyPath().toString().equals("status"))).isTrue();
        }
    }

    // ==================== category Validation Tests ====================

    @Nested
    @DisplayName("category Validation")
    class CategoryValidationTests {

        @Test
        @DisplayName("should fail validation for null category - @NotNull")
        void shouldFailValidationForNullCategory() {
            ticket.setCategory(null);
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("category"))).isTrue();
        }

        @ParameterizedTest
        @EnumSource(TicketCategory.class)
        @DisplayName("should pass validation for all valid category values")
        void shouldPassValidationForAllValidCategoryValues(TicketCategory category) {
            ticket.setCategory(category);
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations.stream()
                    .noneMatch(v -> v.getPropertyPath().toString().equals("category"))).isTrue();
        }
    }

    // ==================== ticketReference Validation Tests ====================

    @Nested
    @DisplayName("ticketReference Validation")
    class TicketReferenceValidationTests {

        @Test
        @DisplayName("should pass validation for valid ticketReference")
        void shouldPassValidationForValidTicketReference() {
            ticket.setTicketReference("TKT-12345");
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }

        @ParameterizedTest
        @NullAndEmptySource
        @ValueSource(strings = {"   ", "\t", "\n"})
        @DisplayName("should fail validation for blank ticketReference - @NotBlank")
        void shouldFailValidationForBlankTicketReference(String reference) {
            ticket.setTicketReference(reference);
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("ticketReference"))).isTrue();
        }

        @Test
        @DisplayName("should fail validation for ticketReference exceeding 32 characters - @Size(max=32)")
        void shouldFailValidationForTicketReferenceExceeding32Characters() {
            // Widened from 20 for the pentest 3.3 format CIO-yyyyMMdd-XXXXXXXX (21 chars).
            ticket.setTicketReference("A".repeat(33));
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("ticketReference"))).isTrue();
        }

        @Test
        @DisplayName("should pass validation for ticketReference with exactly 32 characters")
        void shouldPassValidationForTicketReferenceWithExactly32Characters() {
            ticket.setTicketReference("A".repeat(32));
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }
    }

    // ==================== updaterId Validation Tests ====================

    @Nested
    @DisplayName("updaterId Validation")
    class UpdaterIdValidationTests {

        @Test
        @DisplayName("should pass validation for valid updaterId")
        void shouldPassValidationForValidUpdaterId() {
            ticket.setUpdaterId("admin-user-123");
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }

        @Test
        @DisplayName("should pass validation for null updaterId (optional field)")
        void shouldPassValidationForNullUpdaterId() {
            ticket.setUpdaterId(null);
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }

        @Test
        @DisplayName("should fail validation for updaterId exceeding 255 characters - @Size(max=255)")
        void shouldFailValidationForUpdaterIdExceeding255Characters() {
            ticket.setUpdaterId("A".repeat(256));
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("updaterId"))).isTrue();
        }

        @Test
        @DisplayName("should pass validation for updaterId with exactly 255 characters")
        void shouldPassValidationForUpdaterIdWithExactly255Characters() {
            ticket.setUpdaterId("A".repeat(255));
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }
    }

    // ==================== isResolved() Method Tests ====================

    @Nested
    @DisplayName("isResolved() Method")
    class IsResolvedMethodTests {

        @Test
        @DisplayName("should return true for RESOLVED status")
        void shouldReturnTrueForResolvedStatus() {
            ticket.setStatus(TicketStatus.RESOLVED);
            assertThat(ticket.isResolved()).isTrue();
        }

        @Test
        @DisplayName("should return true for CLOSED status")
        void shouldReturnTrueForClosedStatus() {
            ticket.setStatus(TicketStatus.CLOSED);
            assertThat(ticket.isResolved()).isTrue();
        }

        @Test
        @DisplayName("should return false for OPEN status")
        void shouldReturnFalseForOpenStatus() {
            ticket.setStatus(TicketStatus.OPEN);
            assertThat(ticket.isResolved()).isFalse();
        }

        @Test
        @DisplayName("should return false for IN_PROGRESS status")
        void shouldReturnFalseForInProgressStatus() {
            ticket.setStatus(TicketStatus.IN_PROGRESS);
            assertThat(ticket.isResolved()).isFalse();
        }

        @Test
        @DisplayName("should return false for WAITING_FOR_CUSTOMER status")
        void shouldReturnFalseForWaitingForCustomerStatus() {
            ticket.setStatus(TicketStatus.WAITING_FOR_CUSTOMER);
            assertThat(ticket.isResolved()).isFalse();
        }
    }

    // ==================== addResponse() Method Tests ====================

    @Nested
    @DisplayName("addResponse() Bidirectional Relationship")
    class AddResponseMethodTests {

        @Test
        @DisplayName("should add response to ticket and set bidirectional relationship")
        void shouldAddResponseAndSetBidirectionalRelationship() {
            TicketResponse response = new TicketResponse();
            response.setContent("Thank you for contacting us.");

            ticket.addResponse(response);

            assertThat(ticket.getResponses()).hasSize(1);
            assertThat(ticket.getResponses()).contains(response);
            assertThat(response.getTicket()).isSameAs(ticket);
        }

        @Test
        @DisplayName("should handle adding multiple responses")
        void shouldHandleAddingMultipleResponses() {
            TicketResponse response1 = new TicketResponse();
            response1.setContent("First response");

            TicketResponse response2 = new TicketResponse();
            response2.setContent("Second response");

            TicketResponse response3 = new TicketResponse();
            response3.setContent("Third response");

            ticket.addResponse(response1);
            ticket.addResponse(response2);
            ticket.addResponse(response3);

            assertThat(ticket.getResponses()).hasSize(3);
            assertThat(ticket.getResponses()).containsExactly(response1, response2, response3);
            assertThat(response1.getTicket()).isSameAs(ticket);
            assertThat(response2.getTicket()).isSameAs(ticket);
            assertThat(response3.getTicket()).isSameAs(ticket);
        }

        @Test
        @DisplayName("should maintain bidirectional reference after adding response")
        void shouldMaintainBidirectionalReferenceAfterAddingResponse() {
            TicketResponse response = new TicketResponse();
            response.setContent("Response content");

            ticket.addResponse(response);

            // Verify both directions of the relationship
            assertThat(ticket.getResponses().get(0).getTicket()).isSameAs(ticket);
        }
    }

    // ==================== addAttachment() Method Tests ====================

    @Nested
    @DisplayName("addAttachment() Bidirectional Relationship")
    class AddAttachmentMethodTests {

        @Test
        @DisplayName("should add attachment to ticket and set bidirectional relationship")
        void shouldAddAttachmentAndSetBidirectionalRelationship() {
            TicketAttachment attachment = createValidAttachment();

            ticket.addAttachment(attachment);

            assertThat(ticket.getAttachments()).hasSize(1);
            assertThat(ticket.getAttachments()).contains(attachment);
            assertThat(attachment.getTicket()).isSameAs(ticket);
        }

        @Test
        @DisplayName("should handle adding multiple attachments")
        void shouldHandleAddingMultipleAttachments() {
            TicketAttachment attachment1 = createValidAttachment();
            attachment1.setFileName("file1.png");

            TicketAttachment attachment2 = createValidAttachment();
            attachment2.setFileName("file2.pdf");

            TicketAttachment attachment3 = createValidAttachment();
            attachment3.setFileName("file3.txt");

            ticket.addAttachment(attachment1);
            ticket.addAttachment(attachment2);
            ticket.addAttachment(attachment3);

            assertThat(ticket.getAttachments()).hasSize(3);
            assertThat(attachment1.getTicket()).isSameAs(ticket);
            assertThat(attachment2.getTicket()).isSameAs(ticket);
            assertThat(attachment3.getTicket()).isSameAs(ticket);
        }

        @Test
        @DisplayName("should maintain bidirectional reference after adding attachment")
        void shouldMaintainBidirectionalReferenceAfterAddingAttachment() {
            TicketAttachment attachment = createValidAttachment();

            ticket.addAttachment(attachment);

            // Verify both directions of the relationship
            assertThat(ticket.getAttachments().get(0).getTicket()).isSameAs(ticket);
        }

        private TicketAttachment createValidAttachment() {
            TicketAttachment attachment = new TicketAttachment();
            attachment.setFileName("screenshot.png");
            attachment.setContentType("image/png");
            attachment.setFileUrl("https://storage.example.com/screenshot.png");
            attachment.setFileSize(1024L);
            return attachment;
        }
    }

    // ==================== Default Values Tests ====================

    @Nested
    @DisplayName("Default Values")
    class DefaultValuesTests {

        @Test
        @DisplayName("should have default status as OPEN")
        void shouldHaveDefaultStatusAsOpen() {
            SupportTicket newTicket = new SupportTicket();
            assertThat(newTicket.getStatus()).isEqualTo(TicketStatus.OPEN);
        }

        @Test
        @DisplayName("should have empty responses list by default")
        void shouldHaveEmptyResponsesListByDefault() {
            SupportTicket newTicket = new SupportTicket();
            assertThat(newTicket.getResponses()).isNotNull();
            assertThat(newTicket.getResponses()).isEmpty();
        }

        @Test
        @DisplayName("should have empty attachments list by default")
        void shouldHaveEmptyAttachmentsListByDefault() {
            SupportTicket newTicket = new SupportTicket();
            assertThat(newTicket.getAttachments()).isNotNull();
            assertThat(newTicket.getAttachments()).isEmpty();
        }

        @Test
        @DisplayName("should have null values for optional fields by default")
        void shouldHaveNullValuesForOptionalFieldsByDefault() {
            SupportTicket newTicket = new SupportTicket();
            assertThat(newTicket.getId()).isNull();
            assertThat(newTicket.getUser()).isNull();
            assertThat(newTicket.getTechnicalDescription()).isNull();
            assertThat(newTicket.getIpAddress()).isNull();
            assertThat(newTicket.getAdminAssignee()).isNull();
            assertThat(newTicket.getCreatedTime()).isNull();
            assertThat(newTicket.getLastUpdateTime()).isNull();
            assertThat(newTicket.getResolvedTime()).isNull();
            assertThat(newTicket.getUpdaterId()).isNull();
        }
    }

    // ==================== Complete Validation Test ====================

    @Nested
    @DisplayName("Complete Validation")
    class CompleteValidationTests {

        @Test
        @DisplayName("should pass validation for fully valid ticket")
        void shouldPassValidationForFullyValidTicket() {
            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(ticket);
            assertThat(violations).isEmpty();
        }

        @Test
        @DisplayName("should report all violations for completely invalid ticket")
        void shouldReportAllViolationsForCompletelyInvalidTicket() {
            SupportTicket invalidTicket = new SupportTicket();
            invalidTicket.setContactEmail(null);
            invalidTicket.setSubject(null);
            invalidTicket.setDescription(null);
            invalidTicket.setStatus(null);
            invalidTicket.setCategory(null);
            invalidTicket.setTicketReference(null);

            Set<ConstraintViolation<SupportTicket>> violations = validator.validate(invalidTicket);
            assertThat(violations).hasSizeGreaterThanOrEqualTo(5);
        }
    }

    // ==================== version (optimistic locking) Tests ====================

    @Nested
    @DisplayName("version")
    class VersionTests {

        @Test
        @DisplayName("should be null before persistence")
        void shouldBeNullBeforePersistence() {
            assertThat(ticket.getVersion()).isNull();
        }

        @Test
        @DisplayName("should get and set version")
        void shouldGetAndSetVersion() {
            ticket.setVersion(5L);

            assertThat(ticket.getVersion()).isEqualTo(5L);
        }
    }
}
