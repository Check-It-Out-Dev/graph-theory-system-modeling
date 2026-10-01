package com.sm.instagram.platform.support.ticket;

import com.sm.instagram.platform.common.exceptions.AuthenticationTranslatableException;
import com.sm.instagram.platform.common.exceptions.ResourceNotFoundException;
import com.sm.instagram.platform.common.ratelimit.RateLimit;
import com.sm.instagram.platform.common.ratelimit.RateLimitKeyType;
import com.sm.instagram.platform.common.ratelimit.RateLimitProfile;
import com.sm.instagram.platform.support.ticket.dtos.*;
import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import com.sm.instagram.platform.support.ticket.models.TicketCategory;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.support.ticket.services.SupportTicketService;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.validation.Valid;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.domain.Sort;
import org.springframework.data.web.PageableDefault;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * REST controller for support ticket management.
 */
@Slf4j
@RestController
@RequestMapping("/support/ticket")
@RateLimit(profile = RateLimitProfile.STANDARD, keyType = RateLimitKeyType.USER_ENDPOINT)
public class SupportTicketController {

    private final SupportTicketService ticketService;

    @Autowired
    public SupportTicketController(SupportTicketService ticketService) {
        this.ticketService = ticketService;
    }

    /**
     * Create a new support ticket.
     *
     * @param dto     The ticket data
     * @param request HTTP request to extract client IP
     * @return Complete ticket DTO with all details
     */
    @PostMapping
    @RateLimit(profile = RateLimitProfile.STANDARD)  // 60 req/min, configurable via RL_STANDARD_* for E2E tests
    public ResponseEntity<SupportTicketDtoOut> createTicket(
            @Valid @RequestBody SupportTicketDtoIn dto,
            HttpServletRequest request) {

        String ipAddress = request.getRemoteAddr();

        // GDPR: Log support ticket creation
        log.info("GDPR: Operation=createSupportTicket, Email={}, Category={}, Purpose=customer_support, DataAccessed=user.email,ticket.content, LegalBasis=legitimate_interest",
                dto.getContactEmail() != null ? dto.getContactEmail().replaceAll("(?<=.{3}).(?=.*@)", "*") : "anonymous",
                dto.getCategory());

        log.info("Creating support ticket from IP: {} - Category: {}, Subject: {}",
                ipAddress, dto.getCategory(), dto.getSubject());
        log.debug("Support ticket creation data: {}", dto);
        long startTime = System.currentTimeMillis();

        SupportTicket ticket = ticketService.createTicket(dto, ipAddress);
        SupportTicketDtoOut ticketDto = ticketService.convertToDto(ticket);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully created support ticket with ID: {} and reference: {} in {}ms",
                ticket.getId(), ticket.getTicketReference(), duration);

        // GDPR: Log successful creation
        log.info("GDPR: Operation=createSupportTicket_SUCCESS, TicketRef={}, DataStored=support.request",
                ticket.getTicketReference());

        return ResponseEntity.ok(ticketDto);
    }

    /**
     * Add attachments to a ticket.
     *
     * @param ticketId    The ticket ID
     * @param attachments List of attachment information
     * @return The created attachments
     */
    @PostMapping("/{ticketId}/attachments")
    public ResponseEntity<List<TicketAttachmentDtoOut>> addTicketAttachments(
            @PathVariable Long ticketId,
            @Valid @RequestBody List<TicketAttachmentDtoIn> attachments) {

        log.info("Adding {} attachments to ticket ID: {}", attachments.size(), ticketId);
        long startTime = System.currentTimeMillis();

        List<TicketAttachmentDtoOut> createdAttachments = ticketService.addTicketAttachments(ticketId, attachments);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully added {} attachments to ticket ID: {} in {}ms",
                createdAttachments.size(), ticketId, duration);

        return ResponseEntity.ok(createdAttachments);
    }

    /**
     * Get complete ticket by reference code and email.
     * Used for anonymous access to tickets.
     *
     * @param reference Ticket reference code
     * @param email     Contact email
     * @return Complete ticket with all details, responses and attachments
     */
    // SECURITY (pentest 3.2/3.3): anonymous by-reference lookup — key the
    // rate limit per client IP (not per user) so it actually throttles an
    // unauthenticated enumeration attempt.
    @RateLimit(profile = RateLimitProfile.STANDARD, keyType = RateLimitKeyType.IP_ENDPOINT)
    @GetMapping("/status")
    public ResponseEntity<SupportTicketDtoOut> getTicketByReference(
            @RequestParam String reference,
            @RequestParam String email) {

        // GDPR: Log ticket retrieval
        log.info("GDPR: Operation=getTicketByReference, Email={}, TicketRef={}, Purpose=support_access, DataAccessed=ticket.details",
                email.replaceAll("(?<=.{3}).(?=.*@)", "*"), reference);

        log.info("Retrieving ticket by reference: {} for email: {}", reference, email);
        long startTime = System.currentTimeMillis();

        SupportTicketDtoOut ticket = ticketService.getTicketByReferenceAndEmail(reference, email);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully retrieved ticket by reference: {} in {}ms", reference, duration);
        log.debug("Retrieved ticket status: {}, responses: {}",
                ticket.getStatus(), ticket.getResponses().size());

        return ResponseEntity.ok(ticket);
    }

    /**
     * Get a ticket via a signed magic-link token (unforgeable, expiring).
     * The token is the authorization — no reference/email required.
     *
     * @param token signed access token from the emailed magic link
     * @return the full ticket details
     */
    // SECURITY (pentest 3.2/3.3): unforgeable token — enumeration is
    // impossible; per-IP rate limit still guards resource abuse.
    @RateLimit(profile = RateLimitProfile.STANDARD, keyType = RateLimitKeyType.IP_ENDPOINT)
    @GetMapping("/access")
    public ResponseEntity<SupportTicketDtoOut> getTicketByAccessToken(@RequestParam String token) {
        log.info("GDPR: Operation=getTicketByAccessToken, Purpose=support_access, DataAccessed=ticket.details");
        return ResponseEntity.ok(ticketService.getTicketByAccessToken(token));
    }

    /**
     * Add a customer response to a ticket.
     *
     * @param reference Ticket reference
     * @param email     Contact email
     * @param dto       Response data
     * @return The created response DTO
     */
    // SECURITY (pentest 3.2): anonymous customer reply — key the rate limit
    // per client IP so an unauthenticated caller can't hammer the endpoint.
    @RateLimit(profile = RateLimitProfile.STANDARD, keyType = RateLimitKeyType.IP_ENDPOINT)
    @PostMapping("/response")
    public ResponseEntity<TicketResponseDtoOut> addCustomerResponse(
            @RequestParam String reference,
            @RequestParam String email,
            @Valid @RequestBody TicketResponseDtoIn dto) {

        // GDPR: Log customer response
        log.info("GDPR: Operation=addCustomerResponse, Email={}, TicketRef={}, Purpose=support_communication, DataAccessed=response.content",
                email.replaceAll("(?<=.{3}).(?=.*@)", "*"), reference);

        log.info("Adding customer response to ticket reference: {} from email: {}", reference, email);
        log.debug("Customer response data: {}", dto);
        long startTime = System.currentTimeMillis();

        TicketResponseDtoOut response = ticketService.addCustomerResponseAndReturnDto(reference, email, dto);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully added customer response to ticket reference: {} in {}ms",
                reference, duration);

        return ResponseEntity.ok(response);
    }

    /**
     * Add attachments to a response.
     *
     * @param responseId  The response ID
     * @param attachments List of attachment information
     * @return The created attachments
     */
    @PostMapping("/response/{responseId}/attachments")
    public ResponseEntity<List<ResponseAttachmentDtoOut>> addResponseAttachments(
            @PathVariable Long responseId,
            @Valid @RequestBody List<ResponseAttachmentDtoIn> attachments) {

        log.info("Adding {} attachments to response ID: {}", attachments.size(), responseId);
        long startTime = System.currentTimeMillis();

        List<ResponseAttachmentDtoOut> createdAttachments =
                ticketService.addResponseAttachments(responseId, attachments);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully added {} attachments to response ID: {} in {}ms",
                createdAttachments.size(), responseId, duration);

        return ResponseEntity.ok(createdAttachments);
    }

    /**
     * Get a ticket by ID.
     * Requires authentication.
     *
     * @param id The ticket ID
     * @return The ticket
     */
    @GetMapping("/{id}")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<SupportTicketDtoOut> getTicketById(@PathVariable Long id) {
        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || auth.getPrincipal() == null) {
            log.error("Authentication not available in security context");
            throw new ResourceNotFoundException("error.auth.not_authenticated");
        }
        String firebaseUid = auth.getName();

        // GDPR: Log authenticated ticket access
        log.info("GDPR: Operation=getTicketById, FirebaseUID={}, TicketID={}, Purpose=support_review, DataAccessed=ticket.fulldetails",
                firebaseUid, id);

        log.info("Retrieving support ticket by ID: {}", id);
        long startTime = System.currentTimeMillis();

        SupportTicketDtoOut dto = ticketService.getTicketById(id, false);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully retrieved support ticket ID: {} in {}ms", id, duration);

        return ResponseEntity.ok(dto);
    }

    /**
     * Get tickets for the currently authenticated user.
     * Allows users to view their own support tickets.
     *
     * @param pageable Pagination info
     * @return Page of tickets belonging to the current user
     */
    @GetMapping("/my-tickets")
    @PreAuthorize("isAuthenticated()")
    public ResponseEntity<Page<SupportTicketDtoOut>> getMyTickets(
            @PageableDefault(size = 20, sort = "createdTime", direction = Sort.Direction.DESC) Pageable pageable) {

        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || auth.getPrincipal() == null) {
            log.error("Authentication not available in security context");
            throw new AuthenticationTranslatableException("error.auth.not_authenticated");
        }
        String userUid = auth.getName();

        // GDPR: Log user's own tickets retrieval
        log.info("GDPR: Operation=getMyTickets, FirebaseUID={}, Purpose=support_self_service, DataAccessed=user.own_tickets",
                userUid);

        log.info("User retrieving their own tickets - page: {}, size: {}",
                pageable.getPageNumber(), pageable.getPageSize());
        long startTime = System.currentTimeMillis();

        Page<SupportTicketDtoOut> tickets = ticketService.getTicketsForCurrentUser(pageable);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully retrieved {} user tickets (total: {}) in {}ms",
                tickets.getNumberOfElements(), tickets.getTotalElements(), duration);

        return ResponseEntity.ok(tickets);
    }

    /**
     * Get tickets with filtering.
     * Admin endpoint.
     *
     * @param status      Filter by status
     * @param category    Filter by category
     * @param searchQuery Search in subject or description
     * @param pageable    Pagination info
     * @return Page of matching tickets
     */
    @GetMapping
    @PreAuthorize("isAuthenticated() && hasAuthority('ADMIN')")
    public ResponseEntity<Page<SupportTicketDtoOut>> getTickets(
            @RequestParam(required = false) TicketStatus status,
            @RequestParam(required = false) TicketCategory category,
            @RequestParam(required = false) String searchQuery,
            @PageableDefault(size = 20) Pageable pageable) {

        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || auth.getPrincipal() == null) {
            log.error("Authentication not available in security context");
            throw new AuthenticationTranslatableException("error.auth.not_authenticated");
        }
        String adminUid = auth.getName();

        // GDPR: Log admin ticket listing
        log.info("GDPR: Operation=getTickets, FirebaseUID={}, Purpose=support_management, DataAccessed=ticket.list, Filters={}_{}",
                adminUid, status, category);

        log.info("Admin retrieving tickets - page: {}, size: {}, status: {}, category: {}, search: {}",
                pageable.getPageNumber(), pageable.getPageSize(), status, category, searchQuery);
        long startTime = System.currentTimeMillis();

        Page<SupportTicketDtoOut> tickets = ticketService.findTickets(status, category, searchQuery, pageable);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully retrieved {} support tickets (total: {}) in {}ms",
                tickets.getNumberOfElements(), tickets.getTotalElements(), duration);

        return ResponseEntity.ok(tickets);
    }

    /**
     * Add an admin response to a ticket.
     * Admin endpoint.
     *
     * @param ticketId The ticket ID
     * @param dto      Response data
     * @return The created response
     */
    @PostMapping("/{ticketId}/admin-response")
    @PreAuthorize("isAuthenticated() && hasAuthority('ADMIN')")
    public ResponseEntity<TicketResponseDtoOut> addAdminResponse(
            @PathVariable Long ticketId,
            @Valid @RequestBody AdminTicketResponseDtoIn dto) {

        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || auth.getPrincipal() == null) {
            log.error("Authentication not available in security context");
            throw new AuthenticationTranslatableException("error.auth.not_authenticated");
        }
        String adminUid = auth.getName();

        // GDPR: Log admin response
        log.info("GDPR: Operation=addAdminResponse, FirebaseUID={}, TicketID={}, Purpose=support_response, DataModified=ticket.responses",
                adminUid, ticketId);

        log.info("Admin adding response to ticket ID: {}", ticketId);
        log.debug("Admin response data: {}", dto);
        long startTime = System.currentTimeMillis();

        TicketResponseDtoOut response = ticketService.addAdminResponse(ticketId, dto);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully added admin response to ticket ID: {} in {}ms", ticketId, duration);

        return ResponseEntity.ok(response);
    }

    /**
     * Update ticket status.
     * Admin endpoint.
     *
     * @param ticketId The ticket ID
     * @param status   The new status
     * @return The updated ticket
     */
    @PatchMapping("/{ticketId}/status")
    @PreAuthorize("isAuthenticated() && hasAuthority('ADMIN')")
    public ResponseEntity<SupportTicketDtoOut> updateTicketStatus(
            @PathVariable Long ticketId,
            @RequestParam TicketStatus status) {

        Authentication auth = SecurityContextHolder.getContext().getAuthentication();
        if (auth == null || auth.getPrincipal() == null) {
            log.error("Authentication not available in security context");
            throw new AuthenticationTranslatableException("error.auth.not_authenticated");
        }
        String adminUid = auth.getName();

        // GDPR: Log status update
        log.info("GDPR: Operation=updateTicketStatus, FirebaseUID={}, TicketID={}, NewStatus={}, Purpose=support_management, DataModified=ticket.status",
                adminUid, ticketId, status);

        log.info("Admin updating ticket ID: {} status to: {}", ticketId, status);
        long startTime = System.currentTimeMillis();

        SupportTicketDtoOut ticket = ticketService.updateTicketStatus(ticketId, status);

        long duration = System.currentTimeMillis() - startTime;
        log.info("Successfully updated ticket ID: {} status to: {} in {}ms",
                ticketId, status, duration);

        return ResponseEntity.ok(ticket);
    }
}
