package com.sm.instagram.platform.appliedopportunities;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.sm.instagram.platform.common.base.UpdaterTracking;
import com.sm.instagram.platform.partnershipopportunities.PartnershipOpportunity;
import com.sm.instagram.platform.user.User;
import jakarta.persistence.CascadeType;
import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.EnumType;
import jakarta.persistence.Enumerated;
import jakarta.persistence.FetchType;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.JoinColumn;
import jakarta.persistence.ManyToOne;
import jakarta.persistence.OneToMany;
import jakarta.persistence.SequenceGenerator;
import jakarta.persistence.Table;
import jakarta.persistence.UniqueConstraint;
import jakarta.persistence.Version;
import jakarta.validation.constraints.NotNull;
import jakarta.validation.constraints.Size;
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
 * JPA entity representing an influencer's application to a campaign.
 *
 * <p>Hibernate / Liquibase contract — must stay in lock-step with the
 * matching changeset:
 * <ul>
 *   <li>Table: {@code applied_opportunity}.</li>
 *   <li>Unique constraint {@code uk_influencer_opportunity}: an influencer
 *       can apply to a given partnership opportunity at most once.</li>
 *   <li>Sequence: {@code applied_opportunity_seq} on schema {@code public},
 *       allocation 50.</li>
 *   <li>Enum columns ({@code opportunityStatus}, {@code rateStatus},
 *       {@code companyRateStatus}) are stored as their {@code @Enumerated
 *       (EnumType.STRING)} names, so renaming any enum constant requires a
 *       data migration.</li>
 *   <li>{@code @Version} drives optimistic locking — concurrent edits by
 *       both sides of the cooperation are common in tests, so the column
 *       must be present and not skipped from updates.</li>
 *   <li>{@code influencer} and {@code partnershipOpportunity} are
 *       {@code @JsonIgnore}d to break the cycle on serialization; consumers
 *       receive the projection DTOs instead.</li>
 * </ul>
 */
@Getter
@Setter
@Entity
@Table(
        name = "applied_opportunity",
        uniqueConstraints = {
                @UniqueConstraint(
                        name = "uk_influencer_opportunity",
                        columnNames = {"influencer_id", "partnership_opportunity_id"})
        }
)
@NoArgsConstructor
@AllArgsConstructor
public class AppliedOpportunity implements UpdaterTracking {

    // ----- identity -----

    @Id
    @GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "applied_opportunity_generator")
    @SequenceGenerator(
            name = "applied_opportunity_generator",
            sequenceName = "applied_opportunity_seq",
            schema = "public",
            allocationSize = 50,
            initialValue = 1
    )
    private Long id;

    // ----- associations -----

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "influencer_id", nullable = false)
    @NotNull(message = "Influencer cannot be blank")
    @JsonIgnore
    private User influencer;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "partnership_opportunity_id", nullable = false)
    @NotNull(message = "Partnership Opportunity cannot be blank")
    @JsonIgnore
    private PartnershipOpportunity partnershipOpportunity;

    @OneToMany(mappedBy = "appliedOpportunity", cascade = CascadeType.ALL, fetch = FetchType.LAZY)
    private List<AppliedOpportunityContent> contentSubmissions = new ArrayList<>();

    // ----- scalars -----

    @Size(max = 500, message = "Note cannot exceed 500 characters")
    private String note;

    private LocalDateTime executionDate;

    // ----- status enums (string-persisted) -----

    @Enumerated(EnumType.STRING)
    @NotNull(message = "Opportunity status cannot be blank")
    private OpportunityStatus opportunityStatus = OpportunityStatus.APPLIED;

    @Enumerated(EnumType.STRING)
    private RateStatus rateStatus = RateStatus.DEFAULT;

    @Enumerated(EnumType.STRING)
    private RateStatus companyRateStatus = RateStatus.DEFAULT;

    // ----- optimistic locking -----

    @Version
    @Column(name = "version")
    private Long version;

    // ----- audit fields -----

    @Column(updatable = false)
    @CreationTimestamp
    private LocalDateTime createdTime;

    @UpdateTimestamp
    private LocalDateTime lastUpdateTime = LocalDateTime.now();

    @Size(max = 255, message = "Firebase User ID cannot exceed 255 characters")
    private String updaterId;

    // ----- collection helpers -----

    /**
     * Add a content submission and back-link it to this opportunity. Use in
     * place of {@code contentSubmissions.add(...)} so the inverse side stays
     * consistent without relying on Hibernate to refresh the entity.
     */
    public void addContentSubmission(AppliedOpportunityContent content) {
        contentSubmissions.add(content);
        content.setAppliedOpportunity(this);
    }

    /**
     * Remove a content submission and clear its back-link. Inverse of
     * {@link #addContentSubmission(AppliedOpportunityContent)}.
     */
    public void removeContentSubmission(AppliedOpportunityContent content) {
        contentSubmissions.remove(content);
        content.setAppliedOpportunity(null);
    }
}
