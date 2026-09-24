package com.sm.instagram.platform.integration.service.supportticket;

import com.sm.instagram.platform.support.ticket.dtos.SupportTicketDtoIn;
import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import com.sm.instagram.platform.support.ticket.models.TicketCategory;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.verifyNoInteractions;

/**
 * Integration tests for SupportTicketService ticket creation operations.
 */
@DisplayName("SupportTicketService Create Operations")
class SupportTicketService_Create_IntegrationTest extends SupportTicketServiceIntegrationTestBase {

    @BeforeEach
    void setUpAuth() {
        authenticateAs(testInfluencer);
    }

    @Nested
    @DisplayName("createTicket()")
    class CreateTicket {

        @Test
        @DisplayName("creates ticket with all required fields")
        void createsTicketWithAllRequiredFields() {
            String email = generateTestEmail();
            SupportTicketDtoIn dto = createTicketDtoIn(email, "Test Subject", "Test Description");
            dto.setCategory(TicketCategory.TECHNICAL_PROBLEM);

            SupportTicket result = supportTicketService.createTicket(dto, TEST_IP_ADDRESS);

            assertThat(result.getId()).isNotNull();
            // For authenticated users, service uses their profile email (security: prevents email spoofing)
            assertThat(result.getContactEmail()).isEqualTo(testInfluencer.getEmail());
            assertThat(result.getSubject()).isEqualTo("Test Subject");
            assertThat(result.getDescription()).isEqualTo("Test Description");
            assertThat(result.getCategory()).isEqualTo(TicketCategory.TECHNICAL_PROBLEM);
            assertThat(result.getStatus()).isEqualTo(TicketStatus.OPEN);
            assertThat(result.getIpAddress()).isEqualTo(TEST_IP_ADDRESS);
        }

        @Test
        @DisplayName("persists the error-report technical description")
        void persistsTechnicalDescription() {
            String email = generateTestEmail();
            SupportTicketDtoIn dto = createTicketDtoIn(email, "Application Error", "Auto-filled");
            dto.setTechnicalDescription("=== ERROR DETAILS ===\nstatus: 500\ntrace: req-42");

            SupportTicket result = supportTicketService.createTicket(dto, TEST_IP_ADDRESS);
            flushAndClear();

            SupportTicket persisted = supportTicketRepository.findById(result.getId()).orElseThrow();
            assertThat(persisted.getTechnicalDescription()).contains("trace: req-42");
        }

        @Test
        @DisplayName("generates unique ticket reference")
        void generatesUniqueTicketReference() {
            String email = generateTestEmail();
            SupportTicketDtoIn dto = createTicketDtoIn(email, "Subject", "Description");

            SupportTicket ticket1 = supportTicketService.createTicket(dto, TEST_IP_ADDRESS);

            String email2 = generateTestEmail();
            SupportTicketDtoIn dto2 = createTicketDtoIn(email2, "Subject 2", "Description 2");
            SupportTicket ticket2 = supportTicketService.createTicket(dto2, TEST_IP_ADDRESS);

            assertThat(ticket1.getTicketReference()).isNotNull();
            assertThat(ticket2.getTicketReference()).isNotNull();
            assertThat(ticket1.getTicketReference()).isNotEqualTo(ticket2.getTicketReference());
        }

        @Test
        @DisplayName("ticket reference starts with CIO prefix")
        void ticketReferenceStartsWithCioPrefix() {
            String email = generateTestEmail();
            SupportTicketDtoIn dto = createTicketDtoIn(email, "Subject", "Description");

            SupportTicket result = supportTicketService.createTicket(dto, TEST_IP_ADDRESS);

            assertThat(result.getTicketReference()).startsWith("CIO-");
        }

        @Test
        @DisplayName("does not send the confirmation email inside the creating transaction")
        void doesNotSendConfirmationEmailInsideTheCreatingTransaction() {
            // The confirmation is sent by SupportTicketCreatedEventListener AFTER_COMMIT,
            // which never fires in this @Transactional (rollback) test context — see
            // NotificationService_AccountActivation_IntegrationTest. The listener itself
            // is covered by SupportTicketCreatedEventListenerUnitTest.
            String email = generateTestEmail();
            SupportTicketDtoIn dto = createTicketDtoIn(email, "Subject", "Description");

            supportTicketService.createTicket(dto, TEST_IP_ADDRESS);

            verifyNoInteractions(emailService);
        }

        @Test
        @DisplayName("associates authenticated user with ticket")
        void associatesAuthenticatedUserWithTicket() {
            authenticateAs(testInfluencer);
            String email = generateTestEmail();
            SupportTicketDtoIn dto = createTicketDtoIn(email, "Subject", "Description");

            SupportTicket result = supportTicketService.createTicket(dto, TEST_IP_ADDRESS);

            assertThat(result.getUser()).isNotNull();
            assertThat(result.getUser().getId()).isEqualTo(testInfluencer.getId());
        }

        @Test
        @DisplayName("sets created time automatically")
        void setsCreatedTimeAutomatically() {
            String email = generateTestEmail();
            SupportTicketDtoIn dto = createTicketDtoIn(email, "Subject", "Description");

            SupportTicket result = supportTicketService.createTicket(dto, TEST_IP_ADDRESS);

            // Flush to database and clear persistence context to ensure @CreationTimestamp is populated
            flushAndClear();

            // Fetch fresh from repository to get Hibernate-populated timestamp
            SupportTicket freshTicket = supportTicketRepository.findById(result.getId()).orElseThrow();
            assertThat(freshTicket.getCreatedTime()).isNotNull();
        }
    }

    @Nested
    @DisplayName("Ticket Categories")
    class TicketCategories {

        @Test
        @DisplayName("can create ticket with ACCOUNT_ISSUE category")
        void canCreateTicketWithAccountIssueCategory() {
            String email = generateTestEmail();
            SupportTicketDtoIn dto = createTicketDtoIn(email, "Subject", "Description", TicketCategory.ACCOUNT_ISSUE);

            SupportTicket result = supportTicketService.createTicket(dto, TEST_IP_ADDRESS);

            assertThat(result.getCategory()).isEqualTo(TicketCategory.ACCOUNT_ISSUE);
        }

        @Test
        @DisplayName("can create ticket with BILLING_PAYMENT category")
        void canCreateTicketWithBillingPaymentCategory() {
            String email = generateTestEmail();
            SupportTicketDtoIn dto = createTicketDtoIn(email, "Subject", "Description", TicketCategory.BILLING_PAYMENT);

            SupportTicket result = supportTicketService.createTicket(dto, TEST_IP_ADDRESS);

            assertThat(result.getCategory()).isEqualTo(TicketCategory.BILLING_PAYMENT);
        }

        @Test
        @DisplayName("can create ticket with FEATURE_REQUEST category")
        void canCreateTicketWithFeatureRequestCategory() {
            String email = generateTestEmail();
            SupportTicketDtoIn dto = createTicketDtoIn(email, "Subject", "Description", TicketCategory.FEATURE_REQUEST);

            SupportTicket result = supportTicketService.createTicket(dto, TEST_IP_ADDRESS);

            assertThat(result.getCategory()).isEqualTo(TicketCategory.FEATURE_REQUEST);
        }

        @Test
        @DisplayName("can create ticket with EARLY_ACCESS_INTEREST category")
        void canCreateTicketWithEarlyAccessInterestCategory() {
            String email = generateTestEmail();
            SupportTicketDtoIn dto = createTicketDtoIn(email, "Subject", "Description", TicketCategory.EARLY_ACCESS_INTEREST);

            SupportTicket result = supportTicketService.createTicket(dto, TEST_IP_ADDRESS);

            assertThat(result.getCategory()).isEqualTo(TicketCategory.EARLY_ACCESS_INTEREST);
        }
    }
}
