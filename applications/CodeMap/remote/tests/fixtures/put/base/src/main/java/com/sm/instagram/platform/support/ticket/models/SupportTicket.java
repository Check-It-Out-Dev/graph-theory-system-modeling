package com.sm.instagram.platform.support.ticket.models;

import com.sm.instagram.platform.common.base.UpdaterTracking;
import com.sm.instagram.platform.user.User;
import jakarta.persistence.*;
import jakarta.validation.constraints.*;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.hibernate.annotations.CreationTimestamp;
import org.hibernate.annotations.UpdateTimestamp;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Entity representing a support ticket in the system.
 * Tickets can be created by both logged-in users and anonymous users.
 */
@Entity
@Table(name = "support_ticket")
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
public class SupportTicket implements UpdaterTracking {

    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "support_ticket_generator")
    @SequenceGenerator(
        name = "support_ticket_generator",
        sequenceName = "support_ticket_seq",
        schema = "public",
        allocationSize = 50,
        initialValue = 1
    )
    private Long id;

    /**
     * Optional reference to the user who created the ticket.
     * Can be null for anonymous tickets.
     */
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id")
    private User user;

    /**
     * Email address for communication.
     * Required for all tickets, even from logged-in users.
     */
    @NotBlank(message = "Contact email is required")
    @Email(message = "Invalid email format")
    @Size(max = 255, message = "Email cannot exceed 255 characters")
    private String contactEmail;

    /**
     * Brief description of the issue.
     */
    @NotBlank(message = "Subject cannot be blank")
    @Size(max = 255, message = "Subject cannot exceed 255 characters")
    private String subject;

    /**
     * Detailed description of the issue.
     */
    @NotBlank(message = "Description cannot be blank")
    @Size(max = 5000, message = "Description cannot exceed 5000 characters")
    @Column(length = 5000)
    private String description;

    /**
     * Technical description containing error logs, parameters, debug information etc.
     * This field is for internal/admin use only and should not be exposed to users on frontend.
     */
    @Size(max = 100000, message = "Technical description cannot exceed 100000 characters")
    @Column(length = 100000)
    private String technicalDescription;

    /**
     * Current status of the ticket.
     */
    @NotNull(message = "Status cannot be null")
    @Enumerated(EnumType.STRING)
    private TicketStatus status = TicketStatus.OPEN;

    /**
     * Category of the ticket for organization and routing.
     */
    @NotNull(message = "Category cannot be null")
    @Enumerated(EnumType.STRING)
    private TicketCategory category;

    /**
     * IP address of the submitter (for rate limiting and spam prevention).
     */
    private String ipAddress;

    /**
     * Unique reference code for the ticket.
     * Used in email communications and for anonymous access.
     */
    @NotBlank(message = "Ticket reference cannot be blank")
    @Size(max = 32, message = "Ticket reference cannot exceed 32 characters")
    private String ticketReference;

    /**
     * Admin user assigned to handle this ticket.
     */
    private String adminAssignee;

    /**
     * Timestamp when the ticket was created.
     */
    @CreationTimestamp
    @Column(nullable = false, updatable = false)
    private LocalDateTime createdTime;

    /**
     * Timestamp when the ticket was last updated.
     */
    @UpdateTimestamp
    @Column(nullable = false)
    private LocalDateTime lastUpdateTime;

    /**
     * Timestamp when the ticket was resolved.
     * Null for unresolved tickets.
     */
    private LocalDateTime resolvedTime;

    /**
     * Response history for this ticket.
     */
    @OneToMany(mappedBy = "ticket", cascade = CascadeType.ALL, orphanRemoval = true)
    @OrderBy("createdTime ASC")
    private List<TicketResponse> responses = new ArrayList<>();

    /**
     * Attachments for this ticket.
     */
    @OneToMany(mappedBy = "ticket", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<TicketAttachment> attachments = new ArrayList<>();

    /**
     * ID of the user who last updated this entity.
     */
    @Size(max = 255, message = "Updater ID cannot exceed 255 characters")
    private String updaterId;

    /**
     * Helper method to add a response to this ticket.
     */
    public void addResponse(TicketResponse response) {
        responses.add(response);
        response.setTicket(this);
    }

    /**
     * Helper method to add an attachment to this ticket.
     */
    public void addAttachment(TicketAttachment attachment) {
        attachments.add(attachment);
        attachment.setTicket(this);
    }

    /**
     * Check if the ticket is in a resolved or closed state.
     */
    public boolean isResolved() {
        return status == TicketStatus.RESOLVED || status == TicketStatus.CLOSED;
    }
}