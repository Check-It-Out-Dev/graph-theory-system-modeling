package com.sm.instagram.platform.support.ticket.event;

import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.dtos.SupportTicketDtoIn;
import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import com.sm.instagram.platform.support.ticket.models.TicketCategory;
import com.sm.instagram.platform.support.ticket.repositories.SupportTicketRepository;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import com.sm.instagram.platform.support.ticket.services.TicketAccessTokenService;
import com.sm.instagram.platform.support.ticket.services.TicketReferenceService;
import com.sm.instagram.platform.user.UserRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Mockito;
import org.mockito.invocation.Invocation;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.springframework.context.ApplicationEventPublisher;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatCode;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Hidden acceptance test of the task ticket-created-after-commit (prompt under test, instance backend-conventions). */
@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
class TicketCreatedAfterCommitAcceptanceTest {

    @Mock SupportTicketRepository ticketRepository;
    @Mock UserRepository userRepository;
    @Mock TicketReferenceService referenceService;
    @Mock EmailService emailService;
    @Mock PermissionUtils permissionUtils;
    @Mock TicketAccessTokenService accessTokenService;
    @Mock ApplicationEventPublisher eventPublisher;
    @InjectMocks SupportTicketService service;
    @InjectMocks SupportTicketCreatedEventListener listener;

    @Test
    void creatingATicketPublishesTheEventAndSendsNoMailInsideTheTransaction() {
        when(permissionUtils.getUserId()).thenReturn(null);
        when(referenceService.generateTicketReference()).thenReturn("CIO-20260923-ABCDEFGH");
        when(ticketRepository.save(any(SupportTicket.class))).thenAnswer(inv -> {
            SupportTicket t = inv.getArgument(0);
            t.setId(77L);
            return t;
        });
        when(accessTokenService.mint(anyLong())).thenReturn("signed-token");

        SupportTicketDtoIn dto = new SupportTicketDtoIn();
        dto.setContactEmail("customer@example.com");
        dto.setSubject("Invoice missing");
        dto.setDescription("The March invoice is missing.");
        dto.setCategory(TicketCategory.values()[0]);

        service.createTicket(dto, "10.0.0.1");

        assertThat(Mockito.mockingDetails(emailService).getInvocations()).isEmpty();
        List<SupportTicketCreatedEvent> events = published(eventPublisher);
        assertThat(events).hasSize(1);
        SupportTicketCreatedEvent event = events.get(0);
        assertThat(event.getTicketId()).isEqualTo(77L);
        assertThat(event.getContactEmail()).isEqualTo("customer@example.com");
        assertThat(event.getTicketReference()).isEqualTo("CIO-20260923-ABCDEFGH");
        assertThat(event.getSubject()).isEqualTo("Invoice missing");
        assertThat(event.getStatusToken()).isEqualTo("signed-token");
        assertThat(event.getLanguage()).isNotBlank();
    }

    @Test
    void theListenerSendsTheConfirmation() {
        listener.onTicketCreated(new SupportTicketCreatedEvent(this, 5L, "a@example.com", "CIO-1", "Subject", "pl", "tok"));
        verify(emailService).sendTicketCreationConfirmation("a@example.com", "CIO-1", "Subject", "pl", "tok");
    }

    @Test
    void aMailFailureDoesNotLeaveTheListener() {
        doThrow(new IllegalStateException("smtp down")).when(emailService)
                .sendTicketCreationConfirmation(anyString(), anyString(), anyString(), anyString(), anyString());
        assertThatCode(() -> listener.onTicketCreated(
                new SupportTicketCreatedEvent(this, 5L, "a@example.com", "CIO-1", "Subject", "pl", "tok")))
                .doesNotThrowAnyException();
    }

    private static List<SupportTicketCreatedEvent> published(Object publisher) {
        List<SupportTicketCreatedEvent> out = new ArrayList<>();
        for (Invocation i : Mockito.mockingDetails(publisher).getInvocations()) {
            if (i.getMethod().getName().equals("publishEvent") && i.getArgument(0) instanceof SupportTicketCreatedEvent e) {
                out.add(e);
            }
        }
        return out;
    }
}
