package com.sm.instagram.platform.support.ticket.services;

import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.common.exceptions.AuthenticationTranslatableException;
import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.common.exceptions.InsufficientPermissionsException;
import com.sm.instagram.platform.common.exceptions.ResourceNotFoundException;
import com.sm.instagram.platform.support.common.EmailService;
import com.sm.instagram.platform.support.ticket.dtos.*;
import com.sm.instagram.platform.support.ticket.models.*;
import com.sm.instagram.platform.support.ticket.repositories.ResponseAttachmentRepository;
import com.sm.instagram.platform.support.ticket.repositories.SupportTicketRepository;
import com.sm.instagram.platform.support.ticket.repositories.TicketAttachmentRepository;
import com.sm.instagram.platform.support.ticket.repositories.TicketResponseRepository;
import com.sm.instagram.platform.storage.service.SignedUrlService;
import com.sm.instagram.platform.user.User;
import com.sm.instagram.platform.user.UserRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.i18n.LocaleContextHolder;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/**
 * Service for managing support tickets.
 */
@Slf4j
@Service
public class SupportTicketService {

    private final SupportTicketRepository ticketRepository;
    private final TicketResponseRepository responseRepository;
    private final UserRepository userRepository;
    private final TicketReferenceService referenceService;
    private final EmailService emailService;
    private final PermissionUtils permissionUtils;
    private final SignedUrlService signedUrlService;
    private final TicketAccessTokenService accessTokenService;

    @Autowired
    public SupportTicketService(
            SupportTicketRepository ticketRepository,
            TicketResponseRepository responseRepository,
            TicketAttachmentRepository ticketAttachmentRepository,
            ResponseAttachmentRepository responseAttachmentRepository,
            UserRepository userRepository,
            TicketReferenceService referenceService,
            EmailService emailService,
            PermissionUtils permissionUtils,
            SignedUrlService signedUrlService,
            TicketAccessTokenService accessTokenService) {
        this.ticketRepository = ticketRepository;
        this.responseRepository = responseRepository;
        this.userRepository = userRepository;
        this.referenceService = referenceService;
        this.emailService = emailService;
        this.permissionUtils = permissionUtils;
        this.signedUrlService = signedUrlService;
        this.accessTokenService = accessTokenService;
    }

    /**
     * Get a ticket's full details from a signed access token (magic link).
     * The signed token IS the authorization — no reference/email needed, and
     * it cannot be forged or enumerated (pentest 3.2/3.3).
     *
     * @param token the signed access token from the magic link
     * @return the full ticket DTO
     * @throws AuthenticationTranslatableException if the token is invalid or expired
     * @throws ResourceNotFoundException           if the ticket no longer exists
     */
    @Transactional(readOnly = true)
    public SupportTicketDtoOut getTicketByAccessToken(String token) {
        Long ticketId = accessTokenService.verify(token)
                .orElseThrow(() -> new AuthenticationTranslatableException("error.auth.invalid_token"));
        SupportTicket ticket = getTicketByIdWithAssociations(ticketId);
        log.info("GDPR: Operation=getTicketByAccessToken, TicketID={}, Purpose=ticket_retrieval, DataAccessed=ticket.fulldetails", ticketId);
        return convertToDto(ticket);
    }

    /**
     * Create a new support ticket.
     *
     * @param dto       The ticket data
     * @param ipAddress The IP address of the requester (for rate limiting)
     * @return The created ticket
     */
    @Transactional
    public SupportTicket createTicket(SupportTicketDtoIn dto, String ipAddress) {
        String currentUserId = permissionUtils.getUserId();

        // Determine contact email - for authenticated users, use their profile email
        String contactEmail = dto.getContactEmail();
        User authenticatedUser = null;

        if (currentUserId != null && !currentUserId.isEmpty()) {
            Optional<User> currentUserOpt = userRepository.findByFirebaseUserId(currentUserId);
            if (currentUserOpt.isPresent()) {
                authenticatedUser = currentUserOpt.get();
                // Use user's profile email if available (security: prevents email spoofing)
                if (authenticatedUser.getEmail() != null && !authenticatedUser.getEmail().isBlank()) {
                    contactEmail = authenticatedUser.getEmail();
                    log.debug("Using authenticated user's profile email for ticket instead of provided email");
                }
            }
        }

        // GDPR: Log ticket creation
        log.info("GDPR: Operation=createTicket, FirebaseUID={}, Email={}, Category={}, Purpose=support_request, DataCreated=ticket.content,user.email",
                currentUserId != null ? currentUserId : "anonymous",
                contactEmail != null ? contactEmail.replaceAll("(?<=.{3}).(?=.*@)", "*") : "none",
                dto.getCategory());

        SupportTicket ticket = new SupportTicket();

        // Set basic ticket info
        ticket.setContactEmail(contactEmail);
        ticket.setSubject(dto.getSubject());
        ticket.setDescription(dto.getDescription());
        ticket.setCategory(dto.getCategory());
        ticket.setStatus(TicketStatus.OPEN);
        ticket.setIpAddress(ipAddress);
        // Error-report dump from the FE autofill flow (admin eyes only on
        // the way back out — see getTicketById). Was silently dropped here
        // before, which made the whole error-report pipeline a no-op.
        ticket.setTechnicalDescription(dto.getTechnicalDescription());

        // Generate a unique reference code
        ticket.setTicketReference(referenceService.generateTicketReference());

        // Set the authenticated user if available
        if (authenticatedUser != null) {
            ticket.setUser(authenticatedUser);
        }

        // Save the ticket
        SupportTicket savedTicket = ticketRepository.save(ticket);

        // GDPR: Log successful creation
        log.info("GDPR: Operation=createTicket_SUCCESS, FirebaseUID={}, TicketRef={}, DataStored=support.ticket",
                currentUserId != null ? currentUserId : "anonymous",
                savedTicket.getTicketReference());

        // Send confirmation email with a one-click magic link (signed access
        // token) so the reporter can reach the ticket without re-entering the
        // reference + email (pentest 3.2/3.3 — the token is the authorization).
        String language = LocaleContextHolder.getLocale().getLanguage();
        String statusToken = accessTokenService.mint(savedTicket.getId());
        emailService.sendTicketCreationConfirmation(
                savedTicket.getContactEmail(),
                savedTicket.getTicketReference(),
                savedTicket.getSubject(),
                language,
                statusToken
        );

        return savedTicket;
    }

    /**
     * Add attachments to a ticket.
     *
     * @param ticketId       The ticket ID
     * @param attachmentsDto List of attachment information
     * @return List of created attachment DTOs
     */
    @Transactional
    public List<TicketAttachmentDtoOut> addTicketAttachments(Long ticketId, List<TicketAttachmentDtoIn> attachmentsDto) {
        SupportTicket ticket = getTicketById(ticketId);

        // Verify permission - user must be admin OR ticket owner
        if (!permissionUtils.isAdmin() &&
                (ticket.getUser() == null ||
                        !permissionUtils.getUserId().equals(ticket.getUser().getFirebaseUserId()))) {
            throw new InsufficientPermissionsException("error.auth.insufficient_permissions");
        }

        List<TicketAttachment> attachments = new ArrayList<>();

        for (TicketAttachmentDtoIn attachmentDto : attachmentsDto) {
            // SECURITY (pentest 3.1 + 2026-06-13 hardening): the client hands
            // over ONLY the tracked uploadId; the server derives URL, name,
            // type and size from its own tracking row after verifying
            // ownership + that the blob landed in our bucket.
            SignedUrlService.ResolvedUpload resolved =
                    signedUrlService.resolveOwnedUpload(permissionUtils.getUserId(), attachmentDto.getUploadId());

            TicketAttachment attachment = new TicketAttachment();
            attachment.setTicket(ticket);
            attachment.setFileName(resolved.filename());
            attachment.setContentType(resolved.contentType());
            attachment.setFileUrl(resolved.publicUrl());
            attachment.setFileSize(resolved.fileSize());

            ticket.addAttachment(attachment);
            attachments.add(attachment);
        }

        ticketRepository.save(ticket);

        return attachments.stream()
                .map(this::convertToAttachmentDto)
                .collect(Collectors.toList());
    }

    /**
     * Get ticket by reference code and email.
     * Used for anonymous access to tickets.
     *
     * @param reference Ticket reference code
     * @param email     Contact email
     * @return Complete ticket DTO with all details
     * @throws ResourceNotFoundException if no ticket is found
     */
    @Transactional(readOnly = true)
    public SupportTicketDtoOut getTicketByReferenceAndEmail(String reference, String email) {
        // GDPR: Log ticket access by reference
        log.info("GDPR: Operation=getTicketByReferenceAndEmail, Email={}, TicketRef={}, Purpose=ticket_retrieval, DataAccessed=ticket.fulldetails",
                email.replaceAll("(?<=.{3}).(?=.*@)", "*"), reference);

        SupportTicket ticket = ticketRepository.findByTicketReferenceAndContactEmail(reference, email)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Ticket"));

        // Fetch with associations since we're converting to DTO
        ticket = getTicketByIdWithAssociations(ticket.getId());

        return convertToDto(ticket);
    }

    /**
     * Get ticket status by reference code and email.
     * Used for anonymous access to tickets.
     *
     * @param reference Ticket reference code
     * @param email     Contact email
     * @return Ticket status response
     * @deprecated Use getTicketByReferenceAndEmail instead for complete ticket information
     */
    @Deprecated
    public TicketStatusResponse getTicketStatus(String reference, String email) {
        Optional<SupportTicket> ticketOpt = ticketRepository.findByTicketReferenceAndContactEmail(reference, email);

        if (ticketOpt.isEmpty()) {
            return new TicketStatusResponse(false, "No ticket found with the provided reference and email.");
        }

        SupportTicket ticket = ticketOpt.get();

        List<TicketResponse> responses = responseRepository.findByTicketOrderByCreatedTimeAsc(ticket);
        boolean hasAdminResponse = responses.stream().anyMatch(TicketResponse::isFromAdmin);

        return new TicketStatusResponse(
                true,
                ticket.getTicketReference(),
                ticket.getSubject(),
                ticket.getStatus(),
                ticket.getStatus().getDisplayName(),
                ticket.getCreatedTime(),
                ticket.getLastUpdateTime(),
                "Ticket found.",
                responses.size(),
                hasAdminResponse
        );
    }

    /**
     * Add a customer response to a ticket and return the response DTO.
     *
     * @param reference Ticket reference code
     * @param email     Contact email
     * @param dto       Response data
     * @return The DTO of the created response
     */
    @Transactional
    public TicketResponseDtoOut addCustomerResponseAndReturnDto(String reference, String email, TicketResponseDtoIn dto) {
        // GDPR: Log customer response
        log.info("GDPR: Operation=addCustomerResponse, Email={}, TicketRef={}, Purpose=support_communication, DataCreated=response.content",
                email.replaceAll("(?<=.{3}).(?=.*@)", "*"), reference);

        // Get the ticket
        SupportTicket ticket = ticketRepository.findByTicketReferenceAndContactEmail(reference, email)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Ticket"));

        // Create the response
        TicketResponse response = new TicketResponse();
        response.setTicket(ticket);
        response.setContent(dto.getContent());
        response.setFromAdmin(false);
        response.setEmailSent(false);

        // If ticket is in WAITING_FOR_CUSTOMER status, change it back to IN_PROGRESS
        if (ticket.getStatus() == TicketStatus.WAITING_FOR_CUSTOMER) {
            ticket.setStatus(TicketStatus.IN_PROGRESS);
        }

        // Update ticket last update time
        ticket.setLastUpdateTime(LocalDateTime.now());

        // Save the response and ticket
        ticket.addResponse(response);
        SupportTicket savedTicket = ticketRepository.save(ticket);

        // Get the saved response with ID from the ticket
        TicketResponse savedResponse = savedTicket.getResponses()
                .stream()
                .filter(r -> r.getContent().equals(dto.getContent()))
                .filter(r -> !r.isFromAdmin())
                .max(java.util.Comparator.comparing(TicketResponse::getCreatedTime))
                .orElseThrow(() -> new BusinessRuleTranslatableException("error.business.invalid_state"));

        return convertToResponseDto(savedResponse);
    }

    /**
     * Add a customer response to a ticket.
     * This is for anonymous users who can only access their tickets via reference code.
     *
     * @param reference Ticket reference code
     * @param email     Contact email
     * @param dto       Response data
     * @return The created response
     */
    @Transactional
    public TicketResponse addCustomerResponse(String reference, String email, TicketResponseDtoIn dto) {
        Optional<SupportTicket> ticketOpt = ticketRepository.findByTicketReferenceAndContactEmail(reference, email);

        if (ticketOpt.isEmpty()) {
            throw new ResourceNotFoundException("error.business.item_not_found", "Ticket");
        }

        SupportTicket ticket = ticketOpt.get();

        // Create the response
        TicketResponse response = new TicketResponse();
        response.setTicket(ticket);
        response.setContent(dto.getContent());
        response.setFromAdmin(false);
        response.setEmailSent(false); // No need to notify admin by email

        // If ticket is in WAITING_FOR_CUSTOMER status, change it back to IN_PROGRESS
        if (ticket.getStatus() == TicketStatus.WAITING_FOR_CUSTOMER) {
            ticket.setStatus(TicketStatus.IN_PROGRESS);
        }

        // Update ticket last update time
        ticket.setLastUpdateTime(LocalDateTime.now());

        // Save the response and ticket
        ticket.addResponse(response);
        ticketRepository.save(ticket);

        return response;
    }

    /**
     * Add attachments to a response.
     *
     * @param responseId     The response ID
     * @param attachmentsDto List of attachment information
     * @return List of created attachment DTOs
     */
    @Transactional
    public List<ResponseAttachmentDtoOut> addResponseAttachments(Long responseId, List<ResponseAttachmentDtoIn> attachmentsDto) {
        TicketResponse response = responseRepository.findById(responseId)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Response"));

        // Verify permission - user must be admin OR ticket owner
        if (!permissionUtils.isAdmin() &&
                (response.getTicket().getUser() == null ||
                        !permissionUtils.getUserId().equals(response.getTicket().getUser().getFirebaseUserId()))) {
            throw new InsufficientPermissionsException("error.auth.insufficient_permissions");
        }

        List<ResponseAttachment> attachments = new ArrayList<>();

        for (ResponseAttachmentDtoIn attachmentDto : attachmentsDto) {
            // SECURITY (pentest 3.1 + 2026-06-13 hardening): the client hands
            // over ONLY the tracked uploadId; the server derives URL, name,
            // type and size from its own tracking row after verifying
            // ownership + that the blob landed in our bucket.
            SignedUrlService.ResolvedUpload resolved =
                    signedUrlService.resolveOwnedUpload(permissionUtils.getUserId(), attachmentDto.getUploadId());

            ResponseAttachment attachment = new ResponseAttachment();
            attachment.setResponse(response);
            attachment.setFileName(resolved.filename());
            attachment.setContentType(resolved.contentType());
            attachment.setFileUrl(resolved.publicUrl());
            attachment.setFileSize(resolved.fileSize());

            response.addAttachment(attachment);
            attachments.add(attachment);
        }

        // Save the response with its new attachments. The `attachments` list
        // holds the exact managed instances we just added — after the save
        // they carry their generated IDs, so no value-based re-filtering
        // against the DTOs is needed (the DTO is uploadId-only anyway).
        responseRepository.save(response);

        return attachments.stream()
                .map(this::convertToResponseAttachmentDto)
                .collect(Collectors.toList());
    }

    /**
     * Add a response from an admin to a ticket.
     *
     * @param ticketId The ticket ID
     * @param dto      Response data
     * @return The created response DTO
     */
    @Transactional
    public TicketResponseDtoOut addAdminResponse(Long ticketId, AdminTicketResponseDtoIn dto) {
        String adminUid = permissionUtils.getUserId();

        // GDPR: Log admin response
        log.info("GDPR: Operation=addAdminResponse, FirebaseUID={}, TicketID={}, AdminName={}, Purpose=support_response, DataCreated=admin.response",
                adminUid, ticketId, dto.getAdminName());

        if (!permissionUtils.isAdmin()) {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    adminUid,
                    "addAdminResponse",
                    "SupportTicket#" + ticketId);
        }

        SupportTicket ticket = getTicketById(ticketId);

        // Create the response
        TicketResponse response = new TicketResponse();
        response.setTicket(ticket);
        response.setContent(dto.getContent());
        response.setFromAdmin(true);
        response.setAdminName(dto.getAdminName());
        response.setEmailSent(false); // Will be set to true after email is sent

        // Update ticket status if requested
        if (dto.getNewStatus() != null && ticket.getStatus().canTransitionTo(dto.getNewStatus())) {
            ticket.setStatus(dto.getNewStatus());

            // If transitioning to RESOLVED or CLOSED, set resolvedTime
            if (dto.getNewStatus() == TicketStatus.RESOLVED || dto.getNewStatus() == TicketStatus.CLOSED) {
                ticket.setResolvedTime(LocalDateTime.now());
            }
        } else if (ticket.getStatus() == TicketStatus.OPEN || ticket.getStatus() == TicketStatus.IN_PROGRESS) {
            // Automatically change to WAITING_FOR_CUSTOMER if not specified otherwise
            ticket.setStatus(TicketStatus.WAITING_FOR_CUSTOMER);
        }

        // Update ticket last update time
        ticket.setLastUpdateTime(LocalDateTime.now());

        // Save the response and ticket
        ticket.addResponse(response);
        SupportTicket savedTicket = ticketRepository.save(ticket);

        // Get the saved response with ID from the ticket
        TicketResponse savedResponse = savedTicket.getResponses()
                .stream()
                .filter(r -> r.getContent().equals(dto.getContent()))
                .filter(r -> r.isFromAdmin())
                .max(java.util.Comparator.comparing(TicketResponse::getCreatedTime))
                .orElseThrow(() -> new BusinessRuleTranslatableException("error.business.invalid_state"));

        // Send email notification if requested
        if (dto.isSendEmail()) {
            // GDPR: Log email notification
            log.info("GDPR: Operation=sendSupportNotification, FirebaseUID={}, TicketID={}, Purpose=customer_notification, DataProcessed=email.content",
                    adminUid, ticketId);

            String language = LocaleContextHolder.getLocale().getLanguage();
            String statusToken = accessTokenService.mint(savedTicket.getId());
            emailService.sendAdminResponseNotification(
                    savedTicket.getContactEmail(),
                    savedTicket.getTicketReference(),
                    savedTicket.getSubject(),
                    savedResponse.getContent(),
                    savedResponse.getAdminName(),
                    language,
                    statusToken
            );

            // Mark email as sent
            savedResponse.setEmailSent(true);
            responseRepository.save(savedResponse);
        }

        // Convert to DTO
        return convertToResponseDto(savedResponse);
    }

    /**
     * Get tickets for the currently authenticated user.
     *
     * @param pageable Pagination info
     * @return Page of tickets belonging to the current user
     */
    @Transactional(readOnly = true)
    public Page<SupportTicketDtoOut> getTicketsForCurrentUser(Pageable pageable) {
        String userId = permissionUtils.getUserId();

        // GDPR: Log user tickets retrieval
        log.info("GDPR: Operation=getTicketsForCurrentUser, FirebaseUID={}, Purpose=support_self_service, DataAccessed=user.tickets",
                userId);

        User user = userRepository.findByFirebaseUserId(userId)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "User"));

        return ticketRepository.findByUserOrderByCreatedTimeDesc(user, pageable)
                .map(this::convertToDto);
    }

    /**
     * Find tickets with filtering.
     *
     * @param status      Filter by status
     * @param category    Filter by category
     * @param searchQuery Search in subject or description
     * @param pageable    Pagination info
     * @return Page of matching tickets
     */
    @Transactional(readOnly = true)
    public Page<SupportTicketDtoOut> findTickets(
            TicketStatus status,
            TicketCategory category,
            String searchQuery,
            Pageable pageable) {

        String adminUid = permissionUtils.getUserId();

        // GDPR: Log ticket search
        log.info("GDPR: Operation=findTickets, FirebaseUID={}, Status={}, Category={}, Purpose=support_management, DataAccessed=ticket.list",
                adminUid, status, category);

        // Build specification based on filter criteria
        Specification<SupportTicket> spec = Specification.where(null);

        if (searchQuery != null && !searchQuery.isEmpty()) {
            spec = spec.and((root, query, criteriaBuilder) ->
                    criteriaBuilder.or(
                            criteriaBuilder.like(criteriaBuilder.lower(root.get("subject")),
                                    "%" + searchQuery.toLowerCase() + "%"),
                            criteriaBuilder.like(criteriaBuilder.lower(root.get("description")),
                                    "%" + searchQuery.toLowerCase() + "%")
                    ));
        }

        if (status != null) {
            spec = spec.and((root, query, criteriaBuilder) ->
                    criteriaBuilder.equal(root.get("status"), status));
        }

        if (category != null) {
            spec = spec.and((root, query, criteriaBuilder) ->
                    criteriaBuilder.equal(root.get("category"), category));
        }

        // Use findAllWithAssociationsFetched to properly load all associations
        Page<SupportTicket> ticketsPage = ticketRepository.findAllWithAssociationsFetched(spec, pageable);

        return ticketsPage.map(this::convertToDto);
    }

    /**
     * Update ticket status.
     *
     * @param ticketId The ticket ID
     * @param status   The new status
     * @return The updated ticket
     */
    @Transactional
    public SupportTicketDtoOut updateTicketStatus(Long ticketId, TicketStatus status) {
        String adminUid = permissionUtils.getUserId();

        // GDPR: Log status update
        log.info("GDPR: Operation=updateTicketStatus, FirebaseUID={}, TicketID={}, NewStatus={}, Purpose=support_management, DataModified=ticket.status",
                adminUid, ticketId, status);

        if (!permissionUtils.isAdmin()) {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    adminUid,
                    "updateTicketStatus",
                    "SupportTicket#" + ticketId);
        }

        SupportTicket ticket = getTicketById(ticketId);

        if (!ticket.getStatus().canTransitionTo(status)) {
            throw new BusinessRuleTranslatableException("error.business.invalid_state");
        }

        ticket.setStatus(status);

        // If transitioning to RESOLVED or CLOSED, set resolvedTime
        if (status == TicketStatus.RESOLVED || status == TicketStatus.CLOSED) {
            ticket.setResolvedTime(LocalDateTime.now());
        }

        // Update last update time
        ticket.setLastUpdateTime(LocalDateTime.now());

        // Save the ticket
        ticketRepository.save(ticket);

        return convertToDto(ticket);
    }

    /**
     * Get a ticket by ID.
     *
     * @param id The ticket ID
     * @return The ticket
     */
    public SupportTicket getTicketById(Long id) {
        return ticketRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Ticket"));
    }

    /**
     * Get a ticket by ID with all associations fetched.
     * Use this when you need to access responses and attachments collections.
     *
     * @param id The ticket ID
     * @return The ticket with initialized collections
     */
    // No @Transactional: a private method is not proxied, so readOnly never took effect here.
    // The fetch-join does the work this needed; the caller's transaction is the one in force.
    private SupportTicket getTicketByIdWithAssociations(Long id) {
        return ((SupportTicketRepository) ticketRepository).findByIdWithAssociationsFetched(id)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Ticket"));
    }

    /**
     * Get a ticket by ID as DTO.
     * Includes permission check.
     *
     * @param id        The ticket ID
     * @param adminOnly Whether only admins can access this ticket
     * @return The ticket DTO
     */
    @Transactional(readOnly = true)
    public SupportTicketDtoOut getTicketById(Long id, boolean adminOnly) {
        String userId = permissionUtils.getUserId();

        // GDPR: Log ticket access
        log.info("GDPR: Operation=getTicketById, FirebaseUID={}, TicketID={}, AdminOnly={}, Purpose=ticket_review, DataAccessed=ticket.details",
                userId, id, adminOnly);

        SupportTicket ticket = getTicketById(id);

        // Check permissions
        if (adminOnly && !permissionUtils.isAdmin()) {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    userId,
                    "getTicketById",
                    "SupportTicket#" + id);
        } else if (!adminOnly && !permissionUtils.isAdmin() &&
                (ticket.getUser() == null || !permissionUtils.isUserOwner(ticket.getUser()))) {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    userId,
                    "getTicketById",
                    "SupportTicket#" + id);
        }

        // Fetch with associations since we're converting to DTO
        ticket = getTicketByIdWithAssociations(ticket.getId());

        SupportTicketDtoOut dto = convertToDto(ticket);

        // The stored error-report dump is admin eyes only. The entity keeps
        // it for every ticket, but only admins reviewing a ticket get it in
        // the response — ticket owners and the public reference+email lookup
        // never see it.
        if (permissionUtils.isAdmin()) {
            dto.setTechnicalDescription(ticket.getTechnicalDescription());
        }

        return dto;
    }

    /**
     * Convert a SupportTicket to SupportTicketDtoOut.
     *
     * @param ticket The ticket
     * @return The ticket DTO
     */
    public SupportTicketDtoOut convertToDto(SupportTicket ticket) {
        SupportTicketDtoOut dto = new SupportTicketDtoOut();

        dto.setId(ticket.getId());
        dto.setContactEmail(ticket.getContactEmail());
        dto.setSubject(ticket.getSubject());
        dto.setDescription(ticket.getDescription());
        dto.setStatus(ticket.getStatus());
        dto.setStatusDisplay(ticket.getStatus().getDisplayName());
        dto.setCategory(ticket.getCategory());
        dto.setCategoryDisplay(ticket.getCategory().getDisplayName());
        dto.setTicketReference(ticket.getTicketReference());
        dto.setAdminAssignee(ticket.getAdminAssignee());
        dto.setCreatedTime(ticket.getCreatedTime());
        dto.setLastUpdateTime(ticket.getLastUpdateTime());
        dto.setResolvedTime(ticket.getResolvedTime());
        dto.setResolved(ticket.isResolved());

        // Convert responses
        List<TicketResponseDtoOut> responseDtos = ticket.getResponses().stream()
                .map(this::convertToResponseDto)
                .toList();
        dto.setResponses(responseDtos);

        // Convert attachments
        List<TicketAttachmentDtoOut> attachmentDtos = ticket.getAttachments().stream()
                .map(this::convertToAttachmentDto)
                .toList();
        dto.setAttachments(attachmentDtos);

        return dto;
    }

    /**
     * Convert a TicketResponse to TicketResponseDtoOut.
     *
     * @param response The response
     * @return The response DTO
     */
    public TicketResponseDtoOut convertToResponseDto(TicketResponse response) {
        TicketResponseDtoOut dto = new TicketResponseDtoOut();

        dto.setId(response.getId());
        dto.setTicketId(response.getTicket().getId());
        dto.setContent(response.getContent());
        dto.setFromAdmin(response.isFromAdmin());
        dto.setAdminName(response.getAdminName());
        dto.setCreatedTime(response.getCreatedTime());

        // Convert attachments
        List<ResponseAttachmentDtoOut> attachmentDtos = response.getAttachments().stream()
                .map(this::convertToResponseAttachmentDto)
                .toList();
        dto.setAttachments(attachmentDtos);

        return dto;
    }

    /**
     * Convert a TicketAttachment to TicketAttachmentDtoOut.
     *
     * @param attachment The attachment
     * @return The attachment DTO
     */
    private TicketAttachmentDtoOut convertToAttachmentDto(TicketAttachment attachment) {
        TicketAttachmentDtoOut dto = new TicketAttachmentDtoOut();

        dto.setId(attachment.getId());
        dto.setTicketId(attachment.getTicket().getId());
        dto.setFileName(attachment.getFileName());
        dto.setContentType(attachment.getContentType());
        dto.setFileSize(attachment.getFileSize());
        dto.setUploadTime(attachment.getUploadTime());
        dto.setDownloadUrl(attachment.getFileUrl());

        return dto;
    }

    /**
     * Convert a ResponseAttachment to ResponseAttachmentDtoOut.
     *
     * @param attachment The attachment
     * @return The attachment DTO
     */
    private ResponseAttachmentDtoOut convertToResponseAttachmentDto(ResponseAttachment attachment) {
        ResponseAttachmentDtoOut dto = new ResponseAttachmentDtoOut();

        dto.setId(attachment.getId());
        dto.setResponseId(attachment.getResponse().getId());
        dto.setFileName(attachment.getFileName());
        dto.setContentType(attachment.getContentType());
        dto.setFileSize(attachment.getFileSize());
        dto.setUploadTime(attachment.getUploadTime());
        dto.setDownloadUrl(attachment.getFileUrl());

        return dto;
    }
}