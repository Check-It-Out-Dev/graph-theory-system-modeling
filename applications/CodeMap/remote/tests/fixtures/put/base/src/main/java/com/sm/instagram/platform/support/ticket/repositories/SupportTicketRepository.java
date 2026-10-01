package com.sm.instagram.platform.support.ticket.repositories;

import com.sm.instagram.platform.common.base.BaseRepository;
import com.sm.instagram.platform.support.ticket.models.SupportTicket;
import com.sm.instagram.platform.support.ticket.models.TicketCategory;
import com.sm.instagram.platform.support.ticket.models.TicketStatus;
import com.sm.instagram.platform.user.User;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

/**
 * Repository for SupportTicket entities.
 */
@Repository
public interface SupportTicketRepository extends BaseRepository<SupportTicket, Long>, SupportTicketRepositoryCustom {

    /**
     * Find a ticket by its reference code and contact email.
     * Used for anonymous access to tickets.
     *
     * @param ticketReference The unique ticket reference code
     * @param contactEmail The email address associated with the ticket
     * @return Optional containing the ticket if found
     */
    Optional<SupportTicket> findByTicketReferenceAndContactEmail(String ticketReference, String contactEmail);

    /**
     * Whether a ticket with the given reference already exists. Used to
     * guarantee uniqueness of freshly generated references (pentest 3.3).
     *
     * @param ticketReference The reference code to check
     * @return true if a ticket with this reference exists
     */
    boolean existsByTicketReference(String ticketReference);

    /**
     * Find all tickets belonging to a user.
     *
     * @param user The user who created the tickets
     * @return List of tickets
     */
    List<SupportTicket> findByUser(User user);

    /**
     * Find all tickets belonging to a user with pagination, ordered by creation time descending.
     *
     * @param user The user who created the tickets
     * @param pageable Pagination information
     * @return Page of tickets
     */
    Page<SupportTicket> findByUserOrderByCreatedTimeDesc(User user, Pageable pageable);

    /**
     * Find all tickets by status.
     *
     * @param status The ticket status
     * @param pageable Pagination information
     * @return Page of tickets
     */
    Page<SupportTicket> findByStatus(TicketStatus status, Pageable pageable);

    /**
     * Find tickets by category.
     *
     * @param category The ticket category
     * @param pageable Pagination information
     * @return Page of tickets
     */
    Page<SupportTicket> findByCategory(TicketCategory category, Pageable pageable);

    /**
     * Find tickets by status and category.
     *
     * @param status The ticket status
     * @param category The ticket category
     * @param pageable Pagination information
     * @return Page of tickets
     */
    Page<SupportTicket> findByStatusAndCategory(TicketStatus status, TicketCategory category, Pageable pageable);

    /**
     * Find tickets assigned to a specific admin.
     *
     * @param adminAssignee The admin username
     * @param pageable Pagination information
     * @return Page of tickets
     */
    Page<SupportTicket> findByAdminAssignee(String adminAssignee, Pageable pageable);

    /**
     * Search for tickets containing specific text in subject or description.
     *
     * @param searchTerm The text to search for
     * @param pageable Pagination information
     * @return Page of matching tickets
     */
    @Query("SELECT t FROM SupportTicket t WHERE " +
            "LOWER(t.subject) LIKE LOWER(CONCAT('%', :searchTerm, '%')) OR " +
            "LOWER(t.description) LIKE LOWER(CONCAT('%', :searchTerm, '%'))")
    Page<SupportTicket> search(@Param("searchTerm") String searchTerm, Pageable pageable);
}