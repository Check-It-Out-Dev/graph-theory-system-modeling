package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.common.exceptions.InsufficientPermissionsException;
import com.sm.instagram.platform.common.exceptions.ResourceNotFoundException;
import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.dtos.*;
import com.sm.instagram.platform.support.ticket.models.*;
import com.sm.instagram.platform.support.ticket.repositories.*;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import com.sm.instagram.platform.support.ticket.services.TicketReferenceService;
import com.sm.instagram.platform.support.ticket.services.TicketAccessTokenService;
import com.sm.instagram.platform.storage.service.SignedUrlService;
import com.sm.instagram.platform.user.User;
import com.sm.instagram.platform.user.UserRepository;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;

import java.time.LocalDateTime;
import java.util.*;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;

/**
 * Unit tests for SupportTicketService.
 * Tests business logic, state transitions, and permission checks.
 */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
@DisplayName("SupportTicketService Unit Tests")
class SupportTicketServiceUnitTest {

    @Mock
    private SupportTicketRepository ticketRepository;

    @Mock
    private TicketResponseRepository responseRepository;

    @Mock
    private TicketAttachmentRepository ticketAttachmentRepository;

    @Mock
    private ResponseAttachmentRepository responseAttachmentRepository;

    @Mock
    private UserRepository userRepository;

    @Mock
    private TicketReferenceService referenceService;

    @Mock
    private EmailService emailService;

    @Mock
    private PermissionUtils permissionUtils;

    @Mock
    private SignedUrlService signedUrlService;

    @Mock
    private TicketAccessTokenService accessTokenService;

    @InjectMocks
    private SupportTicketService service;

    private User testUser;
    private SupportTicket testTicket;

    @BeforeEach
    void setUp() {
        testUser = new User();
        testUser.setId(1L);
        testUser.setFirebaseUserId("test-firebase-uid");

        testTicket = new SupportTicket();
        testTicket.setId(1L);
        testTicket.setContactEmail("test@example.com");
        testTicket.setSubject("Test Subject");
        testTicket.setDescription("Test Description");
        testTicket.setCategory(TicketCategory.TECHNICAL_PROBLEM);
        testTicket.setStatus(TicketStatus.OPEN);
        testTicket.setTicketReference("TKT-12345");
        testTicket.setUser(testUser);
        testTicket.setCreatedTime(LocalDateTime.now());
        testTicket.setLastUpdateTime(LocalDateTime.now());
    }

    // ==================== TicketStatus Enum Tests ====================

    @Nested
    @DisplayName("TicketStatus Enum")
    class TicketStatusTests {

        @ParameterizedTest
        @EnumSource(TicketStatus.class)
        @DisplayName("should have display name for all statuses")
        void shouldHaveDisplayNameForAllStatuses(TicketStatus status) {
            // When
            String displayName = status.getDisplayName();

            // Then
            assertThat(displayName).isNotNull();
            assertThat(displayName).isNotEmpty();
        }

        @Test
        @DisplayName("should have correct display names")
        void shouldHaveCorrectDisplayNames() {
            assertThat(TicketStatus.OPEN.getDisplayName()).isEqualTo("Open");
            assertThat(TicketStatus.IN_PROGRESS.getDisplayName()).isEqualTo("In Progress");
            assertThat(TicketStatus.WAITING_FOR_CUSTOMER.getDisplayName()).isEqualTo("Awaiting Your Reply");
            assertThat(TicketStatus.RESOLVED.getDisplayName()).isEqualTo("Resolved");
            assertThat(TicketStatus.CLOSED.getDisplayName()).isEqualTo("Closed");
        }

        @Test
        @DisplayName("OPEN can transition to IN_PROGRESS")
        void openCanTransitionToInProgress() {
            assertThat(TicketStatus.OPEN.canTransitionTo(TicketStatus.IN_PROGRESS)).isTrue();
        }

        @Test
        @DisplayName("OPEN can transition to RESOLVED")
        void openCanTransitionToResolved() {
            assertThat(TicketStatus.OPEN.canTransitionTo(TicketStatus.RESOLVED)).isTrue();
        }

        @Test
        @DisplayName("OPEN can transition to CLOSED")
        void openCanTransitionToClosed() {
            assertThat(TicketStatus.OPEN.canTransitionTo(TicketStatus.CLOSED)).isTrue();
        }

        @Test
        @DisplayName("OPEN cannot transition to WAITING_FOR_CUSTOMER")
        void openCannotTransitionToWaitingForCustomer() {
            assertThat(TicketStatus.OPEN.canTransitionTo(TicketStatus.WAITING_FOR_CUSTOMER)).isFalse();
        }

        @Test
        @DisplayName("IN_PROGRESS can transition to WAITING_FOR_CUSTOMER")
        void inProgressCanTransitionToWaitingForCustomer() {
            assertThat(TicketStatus.IN_PROGRESS.canTransitionTo(TicketStatus.WAITING_FOR_CUSTOMER)).isTrue();
        }

        @Test
        @DisplayName("IN_PROGRESS can transition to RESOLVED")
        void inProgressCanTransitionToResolved() {
            assertThat(TicketStatus.IN_PROGRESS.canTransitionTo(TicketStatus.RESOLVED)).isTrue();
        }

        @Test
        @DisplayName("WAITING_FOR_CUSTOMER can transition to IN_PROGRESS")
        void waitingForCustomerCanTransitionToInProgress() {
            assertThat(TicketStatus.WAITING_FOR_CUSTOMER.canTransitionTo(TicketStatus.IN_PROGRESS)).isTrue();
        }

        @Test
        @DisplayName("RESOLVED can transition to IN_PROGRESS")
        void resolvedCanTransitionToInProgress() {
            assertThat(TicketStatus.RESOLVED.canTransitionTo(TicketStatus.IN_PROGRESS)).isTrue();
        }

        @Test
        @DisplayName("RESOLVED can transition to CLOSED")
        void resolvedCanTransitionToClosed() {
            assertThat(TicketStatus.RESOLVED.canTransitionTo(TicketStatus.CLOSED)).isTrue();
        }

        @Test
        @DisplayName("CLOSED cannot transition to any status")
        void closedCannotTransitionToAnyStatus() {
            for (TicketStatus status : TicketStatus.values()) {
                assertThat(TicketStatus.CLOSED.canTransitionTo(status))
                        .as("CLOSED should not be able to transition to " + status)
                        .isFalse();
            }
        }

        @Test
        @DisplayName("should have 5 statuses")
        void shouldHaveFiveStatuses() {
            assertThat(TicketStatus.values()).hasSize(5);
        }
    }

    // ==================== TicketCategory Enum Tests ====================

    @Nested
    @DisplayName("TicketCategory Enum")
    class TicketCategoryTests {

        @ParameterizedTest
        @EnumSource(TicketCategory.class)
        @DisplayName("should have display name for all categories")
        void shouldHaveDisplayNameForAllCategories(TicketCategory category) {
            // When
            String displayName = category.getDisplayName();

            // Then
            assertThat(displayName).isNotNull();
            assertThat(displayName).isNotEmpty();
        }

        @Test
        @DisplayName("should have correct display names")
        void shouldHaveCorrectDisplayNames() {
            assertThat(TicketCategory.ACCOUNT_ISSUE.getDisplayName()).isEqualTo("Account Issue");
            assertThat(TicketCategory.BILLING_PAYMENT.getDisplayName()).isEqualTo("Billing & Payment");
            assertThat(TicketCategory.TECHNICAL_PROBLEM.getDisplayName()).isEqualTo("Technical Problem");
            assertThat(TicketCategory.FEATURE_REQUEST.getDisplayName()).isEqualTo("Feature Request");
            assertThat(TicketCategory.PARTNERSHIP_ISSUE.getDisplayName()).isEqualTo("Partnership Issue");
            assertThat(TicketCategory.CONTENT_MODERATION.getDisplayName()).isEqualTo("Content Moderation");
            assertThat(TicketCategory.GENERAL_INQUIRY.getDisplayName()).isEqualTo("General Inquiry");
            assertThat(TicketCategory.EARLY_ACCESS_INTEREST.getDisplayName()).isEqualTo("Early Access Interest");
            assertThat(TicketCategory.OTHER.getDisplayName()).isEqualTo("Other");
        }

        @Test
        @DisplayName("should have 9 categories")
        void shouldHaveNineCategories() {
            assertThat(TicketCategory.values()).hasSize(9);
        }
    }

    // ==================== getTicketById Tests ====================

    @Nested
    @DisplayName("getTicketById")
    class GetTicketByIdTests {

        @Test
        @DisplayName("should return ticket when found")
        void shouldReturnTicketWhenFound() {
            // Given
            when(ticketRepository.findById(1L)).thenReturn(Optional.of(testTicket));

            // When
            SupportTicket result = service.getTicketById(1L);

            // Then
            assertThat(result).isNotNull();
            assertThat(result.getId()).isEqualTo(1L);
            assertThat(result.getSubject()).isEqualTo("Test Subject");
        }

        @Test
        @DisplayName("should throw ResourceNotFoundException when not found")
        void shouldThrowResourceNotFoundExceptionWhenNotFound() {
            // Given
            when(ticketRepository.findById(999L)).thenReturn(Optional.empty());

            // When/Then
            assertThatThrownBy(() -> service.getTicketById(999L))
                    .isInstanceOf(ResourceNotFoundException.class)
                    .hasFieldOrPropertyWithValue("messageKey", "error.business.item_not_found");
        }
    }

    // ==================== getTicketByReferenceAndEmail Tests ====================

    @Nested
    @DisplayName("getTicketByReferenceAndEmail")
    class GetTicketByReferenceAndEmailTests {

        @Test
        @DisplayName("should throw ResourceNotFoundException when not found")
        void shouldThrowResourceNotFoundExceptionWhenNotFound() {
            // Given
            when(ticketRepository.findByTicketReferenceAndContactEmail("TKT-99999", "wrong@email.com"))
                    .thenReturn(Optional.empty());

            // When/Then
            assertThatThrownBy(() -> service.getTicketByReferenceAndEmail("TKT-99999", "wrong@email.com"))
                    .isInstanceOf(ResourceNotFoundException.class);
        }
    }

    // ==================== getTicketStatus Tests ====================

    @Nested
    @DisplayName("getTicketStatus (deprecated)")
    class GetTicketStatusTests {

        @Test
        @DisplayName("should return not found status when ticket doesn't exist")
        void shouldReturnNotFoundStatusWhenTicketDoesNotExist() {
            // Given
            when(ticketRepository.findByTicketReferenceAndContactEmail("TKT-99999", "test@example.com"))
                    .thenReturn(Optional.empty());

            // When
            TicketStatusResponse result = service.getTicketStatus("TKT-99999", "test@example.com");

            // Then
            assertThat(result.isFound()).isFalse();
            assertThat(result.getMessage()).contains("No ticket found");
        }

        @Test
        @DisplayName("should return ticket status when found")
        void shouldReturnTicketStatusWhenFound() {
            // Given
            when(ticketRepository.findByTicketReferenceAndContactEmail("TKT-12345", "test@example.com"))
                    .thenReturn(Optional.of(testTicket));
            when(responseRepository.findByTicketOrderByCreatedTimeAsc(testTicket))
                    .thenReturn(Collections.emptyList());

            // When
            TicketStatusResponse result = service.getTicketStatus("TKT-12345", "test@example.com");

            // Then
            assertThat(result.isFound()).isTrue();
            assertThat(result.getTicketReference()).isEqualTo("TKT-12345");
            assertThat(result.getStatus()).isEqualTo(TicketStatus.OPEN);
            assertThat(result.getStatusDisplay()).isEqualTo("Open");
            assertThat(result.getResponseCount()).isEqualTo(0);
            assertThat(result.isHasAdminResponse()).isFalse();
        }

        @Test
        @DisplayName("should indicate admin response when present")
        void shouldIndicateAdminResponseWhenPresent() {
            // Given
            TicketResponse adminResponse = new TicketResponse();
            adminResponse.setFromAdmin(true);
            adminResponse.setContent("Admin response");
            adminResponse.setTicket(testTicket);

            when(ticketRepository.findByTicketReferenceAndContactEmail("TKT-12345", "test@example.com"))
                    .thenReturn(Optional.of(testTicket));
            when(responseRepository.findByTicketOrderByCreatedTimeAsc(testTicket))
                    .thenReturn(List.of(adminResponse));

            // When
            TicketStatusResponse result = service.getTicketStatus("TKT-12345", "test@example.com");

            // Then
            assertThat(result.isHasAdminResponse()).isTrue();
            assertThat(result.getResponseCount()).isEqualTo(1);
        }
    }

    // ==================== Permission Checks Tests ====================

    @Nested
    @DisplayName("Permission Checks")
    class PermissionChecksTests {

        @Test
        @DisplayName("should allow admin to update ticket status")
        void shouldAllowAdminToUpdateTicketStatus() {
            // Given
            when(permissionUtils.getUserId()).thenReturn("admin-uid");
            when(permissionUtils.isAdmin()).thenReturn(true);
            when(ticketRepository.findById(1L)).thenReturn(Optional.of(testTicket));
            when(ticketRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

            // When
            SupportTicketDtoOut result = service.updateTicketStatus(1L, TicketStatus.IN_PROGRESS);

            // Then
            assertThat(result).isNotNull();
            assertThat(result.getStatus()).isEqualTo(TicketStatus.IN_PROGRESS);
        }

        @Test
        @DisplayName("should deny non-admin from updating ticket status")
        void shouldDenyNonAdminFromUpdatingTicketStatus() {
            // Given
            when(permissionUtils.getUserId()).thenReturn("user-uid");
            when(permissionUtils.isAdmin()).thenReturn(false);

            // When/Then
            assertThatThrownBy(() -> service.updateTicketStatus(1L, TicketStatus.IN_PROGRESS))
                    .isInstanceOf(InsufficientPermissionsException.class);
        }

        @Test
        @DisplayName("should deny non-admin from adding admin response")
        void shouldDenyNonAdminFromAddingAdminResponse() {
            // Given
            when(permissionUtils.getUserId()).thenReturn("user-uid");
            when(permissionUtils.isAdmin()).thenReturn(false);

            AdminTicketResponseDtoIn dto = new AdminTicketResponseDtoIn();
            dto.setContent("Admin response");
            dto.setAdminName("Test Admin");

            // When/Then
            assertThatThrownBy(() -> service.addAdminResponse(1L, dto))
                    .isInstanceOf(InsufficientPermissionsException.class);
        }
    }

    // ==================== Status Transition Tests ====================

    @Nested
    @DisplayName("Status Transitions")
    class StatusTransitionTests {

        @Test
        @DisplayName("should allow valid status transition")
        void shouldAllowValidStatusTransition() {
            // Given
            testTicket.setStatus(TicketStatus.OPEN);
            when(permissionUtils.getUserId()).thenReturn("admin-uid");
            when(permissionUtils.isAdmin()).thenReturn(true);
            when(ticketRepository.findById(1L)).thenReturn(Optional.of(testTicket));
            when(ticketRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

            // When
            SupportTicketDtoOut result = service.updateTicketStatus(1L, TicketStatus.IN_PROGRESS);

            // Then
            assertThat(result.getStatus()).isEqualTo(TicketStatus.IN_PROGRESS);
        }

        @Test
        @DisplayName("should reject invalid status transition")
        void shouldRejectInvalidStatusTransition() {
            // Given
            testTicket.setStatus(TicketStatus.OPEN);
            when(permissionUtils.getUserId()).thenReturn("admin-uid");
            when(permissionUtils.isAdmin()).thenReturn(true);
            when(ticketRepository.findById(1L)).thenReturn(Optional.of(testTicket));

            // When/Then - OPEN cannot transition directly to WAITING_FOR_CUSTOMER
            assertThatThrownBy(() -> service.updateTicketStatus(1L, TicketStatus.WAITING_FOR_CUSTOMER))
                    .isInstanceOf(BusinessRuleTranslatableException.class)
                    .hasFieldOrPropertyWithValue("messageKey", "error.business.invalid_state");
        }

        @Test
        @DisplayName("should set resolved time when transitioning to RESOLVED")
        void shouldSetResolvedTimeWhenTransitioningToResolved() {
            // Given
            testTicket.setStatus(TicketStatus.IN_PROGRESS);
            when(permissionUtils.getUserId()).thenReturn("admin-uid");
            when(permissionUtils.isAdmin()).thenReturn(true);
            when(ticketRepository.findById(1L)).thenReturn(Optional.of(testTicket));
            when(ticketRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

            // When
            SupportTicketDtoOut result = service.updateTicketStatus(1L, TicketStatus.RESOLVED);

            // Then
            assertThat(result.getStatus()).isEqualTo(TicketStatus.RESOLVED);
            verify(ticketRepository).save(argThat(ticket ->
                    ticket.getResolvedTime() != null
            ));
        }

        @Test
        @DisplayName("should set resolved time when transitioning to CLOSED")
        void shouldSetResolvedTimeWhenTransitioningToClosed() {
            // Given
            testTicket.setStatus(TicketStatus.OPEN);
            when(permissionUtils.getUserId()).thenReturn("admin-uid");
            when(permissionUtils.isAdmin()).thenReturn(true);
            when(ticketRepository.findById(1L)).thenReturn(Optional.of(testTicket));
            when(ticketRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

            // When
            SupportTicketDtoOut result = service.updateTicketStatus(1L, TicketStatus.CLOSED);

            // Then
            assertThat(result.getStatus()).isEqualTo(TicketStatus.CLOSED);
        }
    }

    // ==================== Customer Response Tests ====================

    @Nested
    @DisplayName("Customer Responses")
    class CustomerResponseTests {

        @Test
        @DisplayName("should add customer response when ticket exists")
        void shouldAddCustomerResponseWhenTicketExists() {
            // Given
            testTicket.setStatus(TicketStatus.WAITING_FOR_CUSTOMER);
            when(ticketRepository.findByTicketReferenceAndContactEmail("TKT-12345", "test@example.com"))
                    .thenReturn(Optional.of(testTicket));
            when(ticketRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

            TicketResponseDtoIn dto = new TicketResponseDtoIn();
            dto.setContent("Customer reply");

            // When
            TicketResponse result = service.addCustomerResponse("TKT-12345", "test@example.com", dto);

            // Then
            assertThat(result).isNotNull();
            assertThat(result.getContent()).isEqualTo("Customer reply");
            assertThat(result.isFromAdmin()).isFalse();
        }

        @Test
        @DisplayName("should change status from WAITING_FOR_CUSTOMER to IN_PROGRESS on customer response")
        void shouldChangeStatusOnCustomerResponse() {
            // Given
            testTicket.setStatus(TicketStatus.WAITING_FOR_CUSTOMER);
            when(ticketRepository.findByTicketReferenceAndContactEmail("TKT-12345", "test@example.com"))
                    .thenReturn(Optional.of(testTicket));
            when(ticketRepository.save(any())).thenAnswer(inv -> {
                SupportTicket saved = inv.getArgument(0);
                return saved;
            });

            TicketResponseDtoIn dto = new TicketResponseDtoIn();
            dto.setContent("Customer reply");

            // When
            service.addCustomerResponse("TKT-12345", "test@example.com", dto);

            // Then
            verify(ticketRepository).save(argThat(ticket ->
                    ticket.getStatus() == TicketStatus.IN_PROGRESS
            ));
        }

        @Test
        @DisplayName("should throw exception when ticket not found for customer response")
        void shouldThrowExceptionWhenTicketNotFoundForCustomerResponse() {
            // Given
            when(ticketRepository.findByTicketReferenceAndContactEmail("TKT-99999", "wrong@email.com"))
                    .thenReturn(Optional.empty());

            TicketResponseDtoIn dto = new TicketResponseDtoIn();
            dto.setContent("Customer reply");

            // When/Then
            assertThatThrownBy(() -> service.addCustomerResponse("TKT-99999", "wrong@email.com", dto))
                    .isInstanceOf(ResourceNotFoundException.class);
        }
    }

    // ==================== Admin Response Tests ====================

    @Nested
    @DisplayName("Admin Responses")
    class AdminResponseTests {

        @Test
        @DisplayName("should auto-set status to WAITING_FOR_CUSTOMER after admin response")
        void shouldAutoSetStatusToWaitingForCustomer() {
            // Given
            testTicket.setStatus(TicketStatus.OPEN);
            when(permissionUtils.getUserId()).thenReturn("admin-uid");
            when(permissionUtils.isAdmin()).thenReturn(true);
            when(ticketRepository.findById(1L)).thenReturn(Optional.of(testTicket));
            when(ticketRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

            AdminTicketResponseDtoIn dto = new AdminTicketResponseDtoIn();
            dto.setContent("Admin reply");
            dto.setAdminName("Admin User");
            dto.setSendEmail(false);

            // When
            service.addAdminResponse(1L, dto);

            // Then
            verify(ticketRepository).save(argThat(ticket ->
                    ticket.getStatus() == TicketStatus.WAITING_FOR_CUSTOMER
            ));
        }

        @Test
        @DisplayName("should send email when requested")
        void shouldSendEmailWhenRequested() {
            // Given
            testTicket.setStatus(TicketStatus.IN_PROGRESS);
            when(permissionUtils.getUserId()).thenReturn("admin-uid");
            when(permissionUtils.isAdmin()).thenReturn(true);
            when(ticketRepository.findById(1L)).thenReturn(Optional.of(testTicket));
            when(ticketRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));
            when(responseRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));
            when(accessTokenService.mint(anyLong())).thenReturn("magic-tok");

            AdminTicketResponseDtoIn dto = new AdminTicketResponseDtoIn();
            dto.setContent("Admin reply");
            dto.setAdminName("Admin User");
            dto.setSendEmail(true);

            // When
            service.addAdminResponse(1L, dto);

            // Then — the minted magic-link token is threaded into the email
            verify(emailService).sendAdminResponseNotification(
                    eq("test@example.com"),
                    eq("TKT-12345"),
                    eq("Test Subject"),
                    eq("Admin reply"),
                    eq("Admin User"),
                    anyString(),
                    eq("magic-tok")
            );
        }

        @Test
        @DisplayName("should apply custom status transition when specified")
        void shouldApplyCustomStatusTransitionWhenSpecified() {
            // Given
            testTicket.setStatus(TicketStatus.IN_PROGRESS);
            when(permissionUtils.getUserId()).thenReturn("admin-uid");
            when(permissionUtils.isAdmin()).thenReturn(true);
            when(ticketRepository.findById(1L)).thenReturn(Optional.of(testTicket));
            when(ticketRepository.save(any())).thenAnswer(inv -> inv.getArgument(0));

            AdminTicketResponseDtoIn dto = new AdminTicketResponseDtoIn();
            dto.setContent("Resolved issue");
            dto.setAdminName("Admin User");
            dto.setNewStatus(TicketStatus.RESOLVED);
            dto.setSendEmail(false);

            // When
            service.addAdminResponse(1L, dto);

            // Then
            verify(ticketRepository).save(argThat(ticket ->
                    ticket.getStatus() == TicketStatus.RESOLVED
            ));
        }
    }

    // ==================== DTO Conversion Tests ====================

    @Nested
    @DisplayName("DTO Conversion")
    class DtoConversionTests {

        @Test
        @DisplayName("should convert ticket to DTO correctly")
        void shouldConvertTicketToDtoCorrectly() {
            // When
            SupportTicketDtoOut dto = service.convertToDto(testTicket);

            // Then
            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getContactEmail()).isEqualTo("test@example.com");
            assertThat(dto.getSubject()).isEqualTo("Test Subject");
            assertThat(dto.getDescription()).isEqualTo("Test Description");
            assertThat(dto.getStatus()).isEqualTo(TicketStatus.OPEN);
            assertThat(dto.getStatusDisplay()).isEqualTo("Open");
            assertThat(dto.getCategory()).isEqualTo(TicketCategory.TECHNICAL_PROBLEM);
            assertThat(dto.getCategoryDisplay()).isEqualTo("Technical Problem");
            assertThat(dto.getTicketReference()).isEqualTo("TKT-12345");
        }

        @Test
        @DisplayName("should convert response to DTO correctly")
        void shouldConvertResponseToDtoCorrectly() {
            // Given
            TicketResponse response = new TicketResponse();
            response.setId(1L);
            response.setTicket(testTicket);
            response.setContent("Test response content");
            response.setFromAdmin(true);
            response.setAdminName("Admin User");
            response.setCreatedTime(LocalDateTime.now());

            // When
            TicketResponseDtoOut dto = service.convertToResponseDto(response);

            // Then
            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getTicketId()).isEqualTo(1L);
            assertThat(dto.getContent()).isEqualTo("Test response content");
            assertThat(dto.isFromAdmin()).isTrue();
            assertThat(dto.getAdminName()).isEqualTo("Admin User");
        }
    }

    // ==================== Entity Tests ====================

    @Nested
    @DisplayName("SupportTicket Entity")
    class SupportTicketEntityTests {

        @Test
        @DisplayName("should create ticket with all required fields")
        void shouldCreateTicketWithAllRequiredFields() {
            // Given
            SupportTicket ticket = new SupportTicket();
            ticket.setContactEmail("user@example.com");
            ticket.setSubject("Help needed");
            ticket.setDescription("I need assistance");
            ticket.setCategory(TicketCategory.ACCOUNT_ISSUE);
            ticket.setStatus(TicketStatus.OPEN);
            ticket.setTicketReference("TKT-00001");

            // Then
            assertThat(ticket.getContactEmail()).isEqualTo("user@example.com");
            assertThat(ticket.getSubject()).isEqualTo("Help needed");
            assertThat(ticket.getDescription()).isEqualTo("I need assistance");
            assertThat(ticket.getCategory()).isEqualTo(TicketCategory.ACCOUNT_ISSUE);
            assertThat(ticket.getStatus()).isEqualTo(TicketStatus.OPEN);
            assertThat(ticket.getTicketReference()).isEqualTo("TKT-00001");
        }

        @Test
        @DisplayName("should track resolved status")
        void shouldTrackResolvedStatus() {
            // Given
            SupportTicket openTicket = new SupportTicket();
            openTicket.setStatus(TicketStatus.OPEN);

            SupportTicket resolvedTicket = new SupportTicket();
            resolvedTicket.setStatus(TicketStatus.RESOLVED);

            // Then
            assertThat(openTicket.isResolved()).isFalse();
            assertThat(resolvedTicket.isResolved()).isTrue();
        }
    }

    // ==================== TicketResponse Entity Tests ====================

    @Nested
    @DisplayName("TicketResponse Entity")
    class TicketResponseEntityTests {

        @Test
        @DisplayName("should create customer response")
        void shouldCreateCustomerResponse() {
            // Given
            TicketResponse response = new TicketResponse();
            response.setTicket(testTicket);
            response.setContent("Customer message");
            response.setFromAdmin(false);

            // Then
            assertThat(response.getContent()).isEqualTo("Customer message");
            assertThat(response.isFromAdmin()).isFalse();
            assertThat(response.getAdminName()).isNull();
        }

        @Test
        @DisplayName("should create admin response")
        void shouldCreateAdminResponse() {
            // Given
            TicketResponse response = new TicketResponse();
            response.setTicket(testTicket);
            response.setContent("Admin message");
            response.setFromAdmin(true);
            response.setAdminName("Admin User");

            // Then
            assertThat(response.getContent()).isEqualTo("Admin message");
            assertThat(response.isFromAdmin()).isTrue();
            assertThat(response.getAdminName()).isEqualTo("Admin User");
        }
    }
}
