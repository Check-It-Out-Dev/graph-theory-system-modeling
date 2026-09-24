package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.dtos.*;
import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import com.sm.instagram.platform.support.ticket.models.TicketCategory;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.repositories.*;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import com.sm.instagram.platform.support.ticket.services.TicketAccessTokenService;
import com.sm.instagram.platform.support.ticket.services.TicketReferenceService;
import com.sm.instagram.platform.storage.service.SignedUrlService;
import com.sm.instagram.platform.user.UserRepository;
import jakarta.persistence.Version;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.orm.ObjectOptimisticLockingFailureException;

import java.lang.reflect.Field;
import java.time.LocalDateTime;
import java.util.Optional;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;

/**
 * Covers optimistic locking on SupportTicket: an admin status change and a
 * customer reply can arrive for the same ticket at the same moment, and the
 * version column must stop the later write from silently discarding the
 * earlier one.
 */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
@DisplayName("SupportTicket optimistic locking")
class SupportTicketVersionUnitTest {

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

    private SupportTicket testTicket;

    @BeforeEach
    void setUp() {
        testTicket = new SupportTicket();
        testTicket.setId(1L);
        testTicket.setContactEmail("test@example.com");
        testTicket.setSubject("Test Subject");
        testTicket.setDescription("Test Description");
        testTicket.setCategory(TicketCategory.TECHNICAL_PROBLEM);
        testTicket.setStatus(TicketStatus.OPEN);
        testTicket.setTicketReference("TKT-12345");
        testTicket.setCreatedTime(LocalDateTime.now());
        testTicket.setLastUpdateTime(LocalDateTime.now());
    }

    @Test
    @DisplayName("version field is mapped with @Version and defaults to null before persist")
    void versionFieldIsMappedForOptimisticLocking() throws NoSuchFieldException {
        Field versionField = SupportTicket.class.getDeclaredField("version");

        assertThat(versionField.getType()).isEqualTo(Long.class);
        assertThat(versionField.isAnnotationPresent(Version.class)).isTrue();
        assertThat(new SupportTicket().getVersion()).isNull();
    }

    @Test
    @DisplayName("updateTicketStatus propagates a version conflict instead of swallowing it")
    void updateTicketStatusPropagatesVersionConflict() {
        when(permissionUtils.getUserId()).thenReturn("admin-uid");
        when(permissionUtils.isAdmin()).thenReturn(true);
        when(ticketRepository.findById(1L)).thenReturn(Optional.of(testTicket));
        when(ticketRepository.save(any(SupportTicket.class)))
                .thenThrow(new ObjectOptimisticLockingFailureException(SupportTicket.class, 1L));

        assertThatThrownBy(() -> service.updateTicketStatus(1L, TicketStatus.IN_PROGRESS))
                .isInstanceOf(ObjectOptimisticLockingFailureException.class);
    }

    @Test
    @DisplayName("addCustomerResponseAndReturnDto propagates a version conflict instead of swallowing it")
    void addCustomerResponsePropagatesVersionConflict() {
        when(ticketRepository.findByTicketReferenceAndContactEmail(eq("TKT-12345"), eq("test@example.com")))
                .thenReturn(Optional.of(testTicket));
        when(ticketRepository.save(any(SupportTicket.class)))
                .thenThrow(new ObjectOptimisticLockingFailureException(SupportTicket.class, 1L));

        TicketResponseDtoIn dto = new TicketResponseDtoIn();
        dto.setContent("Still waiting on this");

        assertThatThrownBy(() -> service.addCustomerResponseAndReturnDto("TKT-12345", "test@example.com", dto))
                .isInstanceOf(ObjectOptimisticLockingFailureException.class);
    }
}
