package com.sm.instagram.platform.appliedopportunities;

import com.sm.instagram.platform.activecooperations.CoopDto;
import com.sm.instagram.platform.activecooperations.CoopFilter;
import com.sm.instagram.platform.activecooperations.Views;
import com.sm.instagram.platform.address.AddressCityOnlyDto;
import com.sm.instagram.platform.address.AddressNoUserDtoOut;
import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.common.base.BaseService;
import com.sm.instagram.platform.common.exceptions.*;
import com.sm.instagram.platform.common.translation.TranslationService;
import com.sm.instagram.platform.common.util.RepositoryResolver;
import com.sm.instagram.platform.common.util.filtering.SpecificationBuilder;
import com.sm.instagram.platform.contenttype.ContentType;
import com.sm.instagram.platform.contenttype.ContentTypeDtoOut;
import com.sm.instagram.platform.currency.CurrencyDtoOut;
import com.sm.instagram.platform.dictionary.DictionaryService;
import com.sm.instagram.platform.notification.event.OpportunityStatusChangedEvent;
import com.sm.instagram.platform.partnershipopportunities.CompensationTypeDtoOut;
import com.sm.instagram.platform.partnershipopportunities.PartnershipOpportunity;
import com.sm.instagram.platform.partnershipopportunities.PartnershipOpportunityPhotoDtoOut;
import com.sm.instagram.platform.partnershipopportunities.PartnershipOpportunitySimpleDtoOut;
import com.sm.instagram.platform.platform.PlatformDto;
import com.sm.instagram.platform.servicetype.ServiceTypeDtoOut;
import com.sm.instagram.platform.user.*;
import com.sm.instagram.platform.userpreferences.UserPreferences;
import com.sm.instagram.platform.userpreferences.UserPreferencesRepository;
import com.sm.instagram.platform.usersocialconnection.ConnectionStatus;
import com.sm.instagram.platform.usersocialconnection.UserSocialConnection;
import com.sm.instagram.platform.usersocialconnection.UserSocialConnectionRepository;
import jakarta.servlet.http.HttpServletRequest;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.hibernate.Hibernate;
import org.modelmapper.ModelMapper;
import org.modelmapper.TypeToken;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.http.converter.json.MappingJacksonValue;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;

import static com.sm.instagram.platform.common.util.RequestContextUtils.getCurrentUser;

@Slf4j
@Service
public class AppliedOpportunityService extends BaseService<AppliedOpportunity, Long, AppliedOpportunityDtoIn> {
    private static final String APPLIED_OPPORTUNITY_NOT_FOUND = "Applied opportunity not found";
    private static final String OPPORTUNITY_STATUS_FIELD = "opportunityStatus";

    private static final String COMPANY_FIELD = "company";
    private static final String INFLUENCER_FIELD = "influencer";

    private final PermissionUtils permissionUtils;
    private final UserRepository userRepository;
    private final UserSocialConnectionRepository userSocialConnectionRepository;
    private final UserPreferencesRepository userPreferencesRepository;
    private final AppliedOpportunityStatusHistoryService statusHistoryService;
    private final DictionaryService dictionaryService;
    private final TranslationService translationService;
    private final HttpServletRequest request;
    private final ApplicationEventPublisher eventPublisher;


    protected AppliedOpportunityService(ApplicationContext applicationContext,
                                        SpecificationBuilder<AppliedOpportunity> specificationBuilder,
                                        AppliedOpportunityRepository repository,
                                        ModelMapper modelMapper,
                                        RepositoryResolver repositoryResolver,
                                        PermissionUtils permissionUtils,
                                        UserRepository userRepository,
                                        UserSocialConnectionRepository userSocialConnectionRepository,
                                        UserPreferencesRepository userPreferencesRepository,
                                        AppliedOpportunityStatusHistoryService statusHistoryService,
                                        DictionaryService dictionaryService,
                                        TranslationService translationService,
                                        HttpServletRequest request,
                                        ApplicationEventPublisher eventPublisher) {
        super(applicationContext, specificationBuilder, repository, modelMapper, repositoryResolver);
        this.permissionUtils = permissionUtils;
        this.userRepository = userRepository;
        this.userSocialConnectionRepository = userSocialConnectionRepository;
        this.userPreferencesRepository = userPreferencesRepository;
        this.statusHistoryService = statusHistoryService;
        this.dictionaryService = dictionaryService;
        this.translationService = translationService;
        this.request = request;
        this.eventPublisher = eventPublisher;
    }

    @Override
    protected AppliedOpportunityService getSelf() {
        return (AppliedOpportunityService) super.getSelf();
    }
    // ===== UTILITY METHODS FOR LAZY LOADING =====

    /**
     * Get locale from Accept-Language header
     */
    private Locale getLocaleFromRequest() {
        String acceptLanguageHeader = request.getHeader("Accept-Language");
        if (acceptLanguageHeader != null && !acceptLanguageHeader.isEmpty()) {
            // Parse the first language from Accept-Language header
            String language = acceptLanguageHeader.split(",")[0].split(";")[0].trim();
            return Locale.forLanguageTag(language);
        }
        return Locale.forLanguageTag("pl"); // Default to Polish
    }

    /**
     * Initializes all lazy associations for an AppliedOpportunity entity.
     * Uses Hibernate.initialize() which is the proper way to force lazy loading
     * without SonarQube warnings about unused return values.
     *
     * @param opportunity The AppliedOpportunity to initialize
     */
    private void initializeLazyAssociations(AppliedOpportunity opportunity) {
        if (opportunity == null) return;

        // Initialize content submissions
        if (opportunity.getContentSubmissions() != null) {
            Hibernate.initialize(opportunity.getContentSubmissions());
        }

        if (opportunity.getInfluencer() != null) {
            Hibernate.initialize(opportunity.getInfluencer());
            if (opportunity.getInfluencer().getAddresses() != null) {
                Hibernate.initialize(opportunity.getInfluencer().getAddresses());
            }
            if (opportunity.getInfluencer().getSocialConnections() != null) {
                Hibernate.initialize(opportunity.getInfluencer().getSocialConnections());
            }
        }
        if (opportunity.getPartnershipOpportunity() != null) {
            initializePartnershipOpportunityAssociations(opportunity.getPartnershipOpportunity());
        }
    }

    /**
     * Initializes lazy associations for a PartnershipOpportunity entity.
     * Extracted to avoid code duplication and improve maintainability.
     *
     * @param partnershipOpportunity The PartnershipOpportunity to initialize
     */
    private void initializePartnershipOpportunityAssociations(
            PartnershipOpportunity partnershipOpportunity) {
        if (partnershipOpportunity == null) return;

        Hibernate.initialize(partnershipOpportunity);

        // Initialize company
        if (partnershipOpportunity.getCompany() != null) {
            Hibernate.initialize(partnershipOpportunity.getCompany());
        }

        // Initialize address and its properties
        if (partnershipOpportunity.getAddress() != null) {
            Hibernate.initialize(partnershipOpportunity.getAddress());
        }

        // Initialize other lazy associations
        if (partnershipOpportunity.getCity() != null) {
            Hibernate.initialize(partnershipOpportunity.getCity());
        }

        if (partnershipOpportunity.getCurrency() != null) {
            Hibernate.initialize(partnershipOpportunity.getCurrency());
        }

        if (partnershipOpportunity.getServiceType() != null) {
            Hibernate.initialize(partnershipOpportunity.getServiceType());
        }

        // Initialize collections
        if (partnershipOpportunity.getPlatforms() != null) {
            Hibernate.initialize(partnershipOpportunity.getPlatforms());
        }

        if (partnershipOpportunity.getContentTypes() != null) {
            Hibernate.initialize(partnershipOpportunity.getContentTypes());
        }

        if (partnershipOpportunity.getPhotos() != null) {
            Hibernate.initialize(partnershipOpportunity.getPhotos());
        }
    }

    // ===== UTILITY METHODS FOR DTO MAPPING =====

    /**
     * Maps an AppliedOpportunity entity to AppliedOpportunityDtoOut.
     * Centralizes DTO mapping logic to avoid code duplication.
     * Must be called within a transaction to avoid LazyInitializationException.
     *
     * @param entity The AppliedOpportunity entity to convert
     * @return AppliedOpportunityDtoOut or null if entity is null
     */
    private AppliedOpportunityDtoOut mapToDto(AppliedOpportunity entity) {
        if (entity == null) return null;

        Locale locale = getLocaleFromRequest();

        AppliedOpportunityDtoOut dto = new AppliedOpportunityDtoOut();
        dto.setId(entity.getId());
        if (entity.getOpportunityStatus() != null) {
            OpportunityStatusDtoOut opportunityStatusDtoOut = OpportunityStatusDtoOut.builder()
                    .value(entity.getOpportunityStatus().name())
                    .label(entity.getOpportunityStatus().getLabel(dictionaryService, locale))
                    .description(entity.getOpportunityStatus().getDescription(dictionaryService, locale))
                    .originalLabel(entity.getOpportunityStatus().name())
                    .colorTheme(entity.getOpportunityStatus().getColorTheme())
                    .icon(entity.getOpportunityStatus().getIcon())
                    .aliases(entity.getOpportunityStatus().getAliases())
                    .possibleTransitions(entity.getOpportunityStatus().getPossibleTransitions().stream()
                            .map(Enum::name).toList())
                    .isTerminal(entity.getOpportunityStatus().isTerminalStatus())
                    .isSuccessful(entity.getOpportunityStatus().isSuccessfulCompletion())
                    .build();
            dto.setOpportunityStatus(opportunityStatusDtoOut);
        }
        dto.setExecutionDate(entity.getExecutionDate());
        dto.setCreatedTime(entity.getCreatedTime());
        dto.setNote(entity.getNote());
        dto.setLastUpdateTime(entity.getLastUpdateTime());
        dto.setUpdater(entity.getUpdaterId());
        dto.setVersion(entity.getVersion());

        dto.setContentSubmissions(mapContentSubmissions(entity));

        // Set influencer based on user type
        dto.setInfluencer(mapInfluencerBasedOnRole(entity.getInfluencer()));

        // Map partnership opportunity with translations
        if (entity.getPartnershipOpportunity() != null) {
            dto.setPartnershipOpportunity(mapPartnershipOpportunityToSimpleDto(entity.getPartnershipOpportunity(), locale));
        }

        // Set rate status based on user role
        mapRateStatusBasedOnRole(entity, dto);

        return dto;
    }

    /**
     * Maps content submissions safely, handling lazy loading issues.
     *
     * @param entity The AppliedOpportunity entity
     * @return List of content submission DTOs or empty list if loading fails
     */
    private List<AppliedOpportunityContentDtoOut> mapContentSubmissions(AppliedOpportunity entity) {
        if (entity.getContentSubmissions() != null) {
            try {
                return modelMapper.map(entity.getContentSubmissions(),
                        new TypeToken<List<AppliedOpportunityContentDtoOut>>() {
                        }.getType());
            } catch (Exception e) {
                log.debug("Could not load content submissions for applied opportunity {}: {}",
                        entity.getId(), e.getMessage());
                return new ArrayList<>();
            }
        }
        return new ArrayList<>();
    }

    /**
     * Maps influencer entity to appropriate DTO based on user role.
     *
     * @param influencer The influencer entity
     * @return Appropriate influencer DTO or null
     */
    @SuppressWarnings("unchecked")
    private <T> T mapInfluencerBasedOnRole(User influencer) {
        if (influencer == null) return null;

        if (permissionUtils.isCompany() || permissionUtils.isAdmin() || permissionUtils.isUserOwner(influencer)) {
            InfluencerForCompanyProfileDto dto = modelMapper.map(influencer, InfluencerForCompanyProfileDto.class);

            // Handle AccountStatus translation manually
            if (influencer.getAccountStatus() != null) {
                AccountStatusDtoOut accountStatusDtoOut = AccountStatusDtoOut.builder()
                        .value(influencer.getAccountStatus().name())
                        .label(influencer.getAccountStatus().getLabel(dictionaryService, getLocaleFromRequest()))
                        .description(influencer.getAccountStatus().getDescription(dictionaryService, getLocaleFromRequest()))
                        .originalLabel(influencer.getAccountStatus().name())
                        .colorTheme(influencer.getAccountStatus().getColorTheme())
                        .icon(influencer.getAccountStatus().getIcon())
                        .isActive(influencer.getAccountStatus().isActive())
                        .canLogin(influencer.getAccountStatus().canLogin())
                        .isTerminal(influencer.getAccountStatus().isTerminal())
                        .build();
                dto.setAccountStatus(accountStatusDtoOut);
            }

            Integer followersCount = getPrimaryFollowerCount(influencer.getId());
            dto.setFollowersCount(followersCount);
            if (influencer.getAddresses() != null && !influencer.getAddresses().isEmpty()) {
                Hibernate.initialize(influencer.getAddresses());
                dto.setAddresses(modelMapper.map(influencer.getAddresses(),
                        new TypeToken<List<AddressCityOnlyDto>>() {
                        }.getType()));
            }

            return (T) dto;
        } else {
            InfluencerPublicProfileDto dto = modelMapper.map(influencer, InfluencerPublicProfileDto.class);

            // Handle AccountStatus translation manually
            if (influencer.getAccountStatus() != null) {
                AccountStatusDtoOut accountStatusDtoOut = AccountStatusDtoOut.builder()
                        .value(influencer.getAccountStatus().name())
                        .label(influencer.getAccountStatus().getLabel(dictionaryService, getLocaleFromRequest()))
                        .description(influencer.getAccountStatus().getDescription(dictionaryService, getLocaleFromRequest()))
                        .originalLabel(influencer.getAccountStatus().name())
                        .colorTheme(influencer.getAccountStatus().getColorTheme())
                        .icon(influencer.getAccountStatus().getIcon())
                        .isActive(influencer.getAccountStatus().isActive())
                        .canLogin(influencer.getAccountStatus().canLogin())
                        .isTerminal(influencer.getAccountStatus().isTerminal())
                        .build();
                dto.setAccountStatus(accountStatusDtoOut);
            }

            Integer followersCount = getPrimaryFollowerCount(influencer.getId());
            dto.setFollowersCount(followersCount);

            return (T) dto;
        }
    }

    /**
     * Maps PartnershipOpportunity to PartnershipOpportunitySimpleDtoOut with proper translations.
     *
     * @param entity The PartnershipOpportunity entity
     * @param locale The locale for translations
     * @return Properly translated PartnershipOpportunitySimpleDtoOut
     */
    private PartnershipOpportunitySimpleDtoOut mapPartnershipOpportunityToSimpleDto(PartnershipOpportunity entity, Locale locale) {
        PartnershipOpportunitySimpleDtoOut dto = new PartnershipOpportunitySimpleDtoOut();

        // Map basic fields
        dto.setId(entity.getId());
        dto.setName(entity.getName());
        dto.setTitle(entity.getTitle());
        dto.setDetails(entity.getDetails());
        dto.setRequirements(entity.getRequirements());
        dto.setFollowersMin(entity.getFollowersMin());
        dto.setFollowersMax(entity.getFollowersMax());
        // Coalesce nullable compensation (legacy rows) to 0 so the int-typed DtoOut contract is preserved.
        dto.setCompensationAmountMin(entity.getCompensationAmountMin() != null ? entity.getCompensationAmountMin() : 0);
        dto.setCompensationAmountMax(entity.getCompensationAmountMax() != null ? entity.getCompensationAmountMax() : 0);
        dto.setCompensationDescription(entity.getCompensationDescription());
        dto.setStartDate(entity.getStartDate());
        dto.setEndDate(entity.getEndDate());
        dto.setActive(entity.isActive());
        dto.setCreatedTime(entity.getCreatedTime());
        dto.setLastUpdateTime(entity.getLastUpdateTime());

        // Map City name
        if (entity.getCity() != null) {
            dto.setCity(entity.getCity().getName());
        }

        // Map CompensationType with translation
        if (entity.getCompensationType() != null) {
            CompensationTypeDtoOut compensationTypeDto = CompensationTypeDtoOut.builder()
                    .value(entity.getCompensationType().name())
                    .label(entity.getCompensationType().getLabel(dictionaryService, locale))
                    .originalLabel(entity.getCompensationType().name())
                    .build();
            dto.setCompensationType(compensationTypeDto);
        }

        // Map Currency with translation
        if (entity.getCurrency() != null) {
            CurrencyDtoOut currencyDto = CurrencyDtoOut.builder()
                    .id(entity.getCurrency().getId())
                    .originalName(entity.getCurrency().getName())
                    .name(translationService.translateCurrency(entity.getCurrency().getIsoCode(), locale))
                    .isoCode(entity.getCurrency().getIsoCode())
                    .sign(entity.getCurrency().getSign())
                    .countryCode(entity.getCurrency().getCountryCode())
                    .displayOrder(entity.getCurrency().getDisplayOrder())
                    .build();
            dto.setCurrency(currencyDto);
        }

        // Map ServiceType with translation
        if (entity.getServiceType() != null) {
            String translatedDescription = entity.getServiceType().getDescription() != null ?
                    translationService.getServiceTypeDescription(entity.getServiceType().getName(), locale) : null;

            ServiceTypeDtoOut serviceTypeDto = ServiceTypeDtoOut.builder()
                    .id(entity.getServiceType().getId())
                    .originalName(entity.getServiceType().getName())
                    .name(translationService.translateServiceType(entity.getServiceType().getName(), locale))
                    .originalDescription(entity.getServiceType().getDescription())
                    .description(translatedDescription != null ? translatedDescription : entity.getServiceType().getDescription())
                    .originalCategory(entity.getServiceType().getCategory())
                    .category(translationService.translateServiceCategory(entity.getServiceType().getCategory(), locale))
                    .build();
            dto.setServiceType(serviceTypeDto);
        }

        // Map Company with AccountStatus translation
        if (entity.getCompany() != null) {
            CompanyPublicProfileDto companyDto = modelMapper.map(entity.getCompany(), CompanyPublicProfileDto.class);

            // Handle AccountStatus translation manually for company
            if (entity.getCompany().getAccountStatus() != null) {
                AccountStatusDtoOut accountStatusDtoOut = AccountStatusDtoOut.builder()
                        .value(entity.getCompany().getAccountStatus().name())
                        .label(entity.getCompany().getAccountStatus().getLabel(dictionaryService, locale))
                        .description(entity.getCompany().getAccountStatus().getDescription(dictionaryService, locale))
                        .originalLabel(entity.getCompany().getAccountStatus().name())
                        .colorTheme(entity.getCompany().getAccountStatus().getColorTheme())
                        .icon(entity.getCompany().getAccountStatus().getIcon())
                        .isActive(entity.getCompany().getAccountStatus().isActive())
                        .canLogin(entity.getCompany().getAccountStatus().canLogin())
                        .isTerminal(entity.getCompany().getAccountStatus().isTerminal())
                        .build();
                companyDto.setAccountStatus(accountStatusDtoOut);
            }

            dto.setCompany(companyDto);
        }

        // Map Photos
        if (entity.getPhotos() != null && !entity.getPhotos().isEmpty()) {
            List<PartnershipOpportunityPhotoDtoOut> photoDtos = entity.getPhotos().stream()
                    .map(photo -> modelMapper.map(photo, PartnershipOpportunityPhotoDtoOut.class))
                    .toList();
            dto.setPhotos(photoDtos);
        }

        // Map Platforms
        if (entity.getPlatforms() != null && !entity.getPlatforms().isEmpty()) {
            Set<PlatformDto> platformDtos = entity.getPlatforms().stream()
                    .map(platform -> modelMapper.map(platform, PlatformDto.class))
                    .collect(Collectors.toSet());
            dto.setPlatforms(platformDtos);
        }

        // Map ContentTypes with translation
        if (entity.getContentTypes() != null && !entity.getContentTypes().isEmpty()) {
            Set<ContentTypeDtoOut> translatedContentTypes = new HashSet<>();
            for (ContentType contentType : entity.getContentTypes()) {
                ContentTypeDtoOut contentTypeDto = ContentTypeDtoOut.builder()
                        .id(contentType.getId())
                        .originalName(contentType.getName())
                        .name(translationService.translateContentType(contentType.getName(), locale))
                        .build();
                translatedContentTypes.add(contentTypeDto);
            }
            dto.setContentTypes(translatedContentTypes);
        }

        // Map Address
        if (entity.getAddress() != null) {
            dto.setAddress(modelMapper.map(entity.getAddress(), AddressNoUserDtoOut.class));
        }

        // Map Updater
        if (entity.getUpdaterId() != null) {
            try {
                User updater = userRepository.findByFirebaseUserId(entity.getUpdaterId())
                        .orElse(null);
                if (updater != null) {
                    dto.setUpdater(updater.getName());
                } else {
                    dto.setUpdater(entity.getUpdaterId());
                }
            } catch (Exception e) {
                dto.setUpdater(entity.getUpdaterId());
            }
        }

        return dto;
    }

    /**
     * Maps rate status fields to DTO based on user role.
     *
     * @param entity The AppliedOpportunity entity
     * @param dto    The DTO to populate
     */
    public void mapRateStatusBasedOnRole(AppliedOpportunity entity, AppliedOpportunityDtoOut dto) {
        Locale locale = getLocaleFromRequest();

        if (permissionUtils.isAdmin()) {
            if (entity.getRateStatus() != null) {
                dto.setRateStatus(RateStatusDtoOut.builder()
                        .value(entity.getRateStatus().name())
                        .label(entity.getRateStatus().getLabel(dictionaryService, locale))
                        .originalLabel(entity.getRateStatus().name())
                        .colorTheme(entity.getRateStatus().getColorTheme())
                        .icon(entity.getRateStatus().getIcon())
                        .build());
            }
            if (entity.getCompanyRateStatus() != null) {
                dto.setCompanyRateStatus(RateStatusDtoOut.builder()
                        .value(entity.getCompanyRateStatus().name())
                        .label(entity.getCompanyRateStatus().getLabel(dictionaryService, locale))
                        .originalLabel(entity.getCompanyRateStatus().name())
                        .colorTheme(entity.getCompanyRateStatus().getColorTheme())
                        .icon(entity.getCompanyRateStatus().getIcon())
                        .build());
            }
        } else if (permissionUtils.isInfluencer()) {
            // Influencer should see THEIR OWN rating (rateStatus) - what they rated the company
            if (entity.getRateStatus() != null) {
                dto.setRateStatus(RateStatusDtoOut.builder()
                        .value(entity.getRateStatus().name())
                        .label(entity.getRateStatus().getLabel(dictionaryService, locale))
                        .originalLabel(entity.getRateStatus().name())
                        .colorTheme(entity.getRateStatus().getColorTheme())
                        .icon(entity.getRateStatus().getIcon())
                        .build());
            }
            // ALSO show company's rating of them (companyRateStatus) - what they received
            if (entity.getCompanyRateStatus() != null) {
                dto.setCompanyRateStatus(RateStatusDtoOut.builder()
                        .value(entity.getCompanyRateStatus().name())
                        .label(entity.getCompanyRateStatus().getLabel(dictionaryService, locale))
                        .originalLabel(entity.getCompanyRateStatus().name())
                        .colorTheme(entity.getCompanyRateStatus().getColorTheme())
                        .icon(entity.getCompanyRateStatus().getIcon())
                        .build());
            }
        } else if (permissionUtils.isCompany()) {
            // Company should see THEIR OWN rating (companyRateStatus) - what they rated the influencer
            if (entity.getCompanyRateStatus() != null) {
                dto.setCompanyRateStatus(RateStatusDtoOut.builder()
                        .value(entity.getCompanyRateStatus().name())
                        .label(entity.getCompanyRateStatus().getLabel(dictionaryService, locale))
                        .originalLabel(entity.getCompanyRateStatus().name())
                        .colorTheme(entity.getCompanyRateStatus().getColorTheme())
                        .icon(entity.getCompanyRateStatus().getIcon())
                        .build());
            }
            // ALSO show influencer's rating of them (rateStatus) - what they received
            if (entity.getRateStatus() != null) {
                dto.setRateStatus(RateStatusDtoOut.builder()
                        .value(entity.getRateStatus().name())
                        .label(entity.getRateStatus().getLabel(dictionaryService, locale))
                        .originalLabel(entity.getRateStatus().name())
                        .colorTheme(entity.getRateStatus().getColorTheme())
                        .icon(entity.getRateStatus().getIcon())
                        .build());
            }
        }
    }

    @Override
    @Transactional(readOnly = true)
    public AppliedOpportunity findById(Long id) {
        String firebaseUid = permissionUtils.getUserId();

        // GDPR: Log opportunity data access
        log.info("GDPR: Operation=findAppliedOpportunityById, FirebaseUID={}, OpportunityID={}, Purpose=opportunity_retrieval, DataAccessed=opportunity.details,influencer.data,company.data",
                firebaseUid, id);

        // Use the custom method with multiple queries to avoid MultipleBagFetchException
        AppliedOpportunity opportunity = ((AppliedOpportunityRepository) repository).findByIdWithAssociationsFetched(id)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Applied opportunity"));

        if (!permissionUtils.canViewAppliedOpportunity(opportunity)) {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    firebaseUid,
                    "findById",
                    "AppliedOpportunity#" + id);
        }

        // Initialize all lazy associations using the utility method
        initializeLazyAssociations(opportunity);

        return opportunity;
    }

    public List<AppliedOpportunity> findByInfluencerUserId(String firebaseUserId) {
        // GDPR: Log influencer opportunities retrieval
        log.info("GDPR: Operation=findByInfluencerUserId, FirebaseUID={}, Purpose=user_opportunities_retrieval, DataAccessed=applied_opportunities.list",
                firebaseUserId);

        List<AppliedOpportunity> opportunities = ((AppliedOpportunityRepository) repository).findAllByInfluencer_FirebaseUserId(firebaseUserId);

        // GDPR: Log data retrieval success
        log.info("GDPR: Operation=findByInfluencerUserId_SUCCESS, FirebaseUID={}, RecordsRetrieved={}",
                firebaseUserId, opportunities.size());

        return opportunities;
    }

    @Override
    @Transactional(readOnly = true)
    public Page<AppliedOpportunity> getDataPagedAndFiltered(Pageable pageable, Map<String, String> filters) {
        String firebaseUid = permissionUtils.getUserId();

        // GDPR: Log paginated data access
        log.info("GDPR: Operation=getDataPagedAndFiltered, FirebaseUID={}, Purpose=opportunities_listing, PageSize={}, PageNumber={}",
                firebaseUid, pageable.getPageSize(), pageable.getPageNumber());

        Specification<AppliedOpportunity> spec = createSpecification(filters);

        if (!permissionUtils.isAdmin()) {
            // Check both role and user type for companies - same as PartnershipOpportunityService
            if (permissionUtils.isCompany() || permissionUtils.isCurrentUserTypeCompany()) {
                // Company users see applications to their opportunities
                spec = spec.and((root, query, criteriaBuilder) ->
                        criteriaBuilder.equal(root.get("partnershipOpportunity").get(COMPANY_FIELD).get("firebaseUserId"), permissionUtils.getUserId()));
            }
            // Check both role and user type for influencers - same as PartnershipOpportunityService
            else if (permissionUtils.isInfluencer() || permissionUtils.isCurrentUserTypeInfluencer()) {
                // Influencer users see their own applications
                User influencer = userRepository.findByFirebaseUserId(permissionUtils.getUserId())
                        .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Influencer"));
                spec = spec.and((root, query, criteriaBuilder) ->
                        criteriaBuilder.equal(root.get(INFLUENCER_FIELD).get("id"), influencer.getId()));
            } else {
                // Unknown user type - show nothing for security
                spec = spec.and((root, query, criteriaBuilder) ->
                        criteriaBuilder.disjunction() // Always false
                );
            }
        }

        // Use the custom method with multiple queries to avoid MultipleBagFetchException
        Page<AppliedOpportunity> page = ((AppliedOpportunityRepository) repository).findAllWithAssociationsFetched(spec, pageable);

        // Initialize lazy associations for all entities in the page using the utility method
        page.getContent().forEach(this::initializeLazyAssociations);

        return page;
    }

    @Override
    public AppliedOpportunity save(AppliedOpportunity appliedOpportunity) {
        String firebaseUid = permissionUtils.getUserId();

        // GDPR: Log opportunity application creation
        log.info("GDPR: Operation=createAppliedOpportunity, FirebaseUID={}, PartnershipOpportunityID={}, Purpose=opportunity_application, DataCreated=application.details, LegalBasis=contract",
                firebaseUid, appliedOpportunity.getPartnershipOpportunity() != null ? appliedOpportunity.getPartnershipOpportunity().getId() : "unknown");

        if (!permissionUtils.canEditAppliedOpportunity(appliedOpportunity))
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    firebaseUid,
                    "save",
                    "AppliedOpportunity");

        // Prevent duplicate applications
        if (appliedOpportunity.getId() == null) { // Only check for new applications
            Long influencerId = appliedOpportunity.getInfluencer().getId();
            Long opportunityId = appliedOpportunity.getPartnershipOpportunity().getId();
            if (((AppliedOpportunityRepository) repository).existsByInfluencerIdAndPartnershipOpportunityId(influencerId, opportunityId)) {
                throw new BusinessRuleTranslatableException("error.business.duplicate_entry", "Application");
            }
        }

        if (!permissionUtils.isAdmin()) {
            User influencer = appliedOpportunity.getInfluencer();
            if (influencer != null && influencer.getUserType() == UserType.INFLUENCER) {
                if (influencer.getSocialConnections() == null || influencer.getSocialConnections().isEmpty()) {
                    throw new InsufficientPermissionsException(
                            "error.auth.insufficient_permissions",
                            firebaseUid,
                            "save",
                            "AppliedOpportunity (no social connections)");
                }

                // NEW: Validate follower range against partnership opportunity requirements
                validateFollowerRange(influencer, appliedOpportunity.getPartnershipOpportunity());
            }
        }
        // Save the entity first
        AppliedOpportunity saved = super.save(appliedOpportunity);
        eventPublisher.publishEvent(new OpportunityStatusChangedEvent(
                this,
                saved,
                null,
                saved.getOpportunityStatus(),
                userRepository.findByFirebaseUserId(getCurrentUser(request)).orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "User")),
                saved.getNote()
        ));
        // GDPR: Log successful creation
        log.info("GDPR: Operation=createAppliedOpportunity_SUCCESS, FirebaseUID={}, AppliedOpportunityID={}, Status={}",
                firebaseUid, saved.getId(), saved.getOpportunityStatus());

        // Log the initial status change (creation)
        statusHistoryService.logStatusChange(
                saved,
                null, // No previous status for new entities
                saved.getOpportunityStatus(),
                "Applied opportunity created"
        );

        return saved;
    }

    @Override
    @Transactional
    public AppliedOpportunity update(Long id, AppliedOpportunityDtoIn dtoIn) {
        if (dtoIn.getInfluencer() == null) {
            throw new ValidationTranslatableException("error.validation.required_field", "influencer");
        }
        var influencer = userRepository.findById(dtoIn.getInfluencer());
        if (influencer.isEmpty()) {
            throw new ResourceNotFoundException("error.business.item_not_found", "Influencer");
        }

        if (!permissionUtils.canEditOpportunity(influencer.get().getFirebaseUserId())) {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    permissionUtils.getUserId(),
                    "update",
                    "AppliedOpportunity#" + id);
        }

        // Use self-injection to call transactional method properly
        AppliedOpportunity existingEntity = getSelf().findById(id);

        // Store original values before mapping to prevent unauthorized changes
        OpportunityStatus originalOpportunityStatus = existingEntity.getOpportunityStatus();
        RateStatus originalRateStatus = existingEntity.getRateStatus();
        RateStatus originalCompanyRateStatus = existingEntity.getCompanyRateStatus();

        modelMapper.map(dtoIn, existingEntity);

        // Track status changes for logging
        OpportunityStatus newOpportunityStatus = originalOpportunityStatus;
        boolean statusChanged = false;

        // If user is not admin, restore original status values to prevent unauthorized changes
        if (!permissionUtils.isAdmin()) {
            existingEntity.setOpportunityStatus(originalOpportunityStatus);

            // Apply permission logic for rating fields
            if (permissionUtils.isInfluencer() &&
                    existingEntity.getInfluencer().getFirebaseUserId().equals(permissionUtils.getUserId())) {
                // Influencers can only update their own rateStatus (rating of company)
                // Restore company rating to prevent unauthorized changes
                existingEntity.setCompanyRateStatus(originalCompanyRateStatus);
            } else if (permissionUtils.isCompany() &&
                    existingEntity.getPartnershipOpportunity().getCompany().getFirebaseUserId().equals(permissionUtils.getUserId())) {
                // Companies can only update companyRateStatus (rating of influencer)
                // Restore influencer rating to prevent unauthorized changes  
                existingEntity.setRateStatus(originalRateStatus);
            } else {
                // Restore both ratings if user has no permission
                existingEntity.setRateStatus(originalRateStatus);
                existingEntity.setCompanyRateStatus(originalCompanyRateStatus);
            }
        } else {
            // Admin can update status fields explicitly
            if (dtoIn.getOpportunityStatus() != null && !dtoIn.getOpportunityStatus().equals(originalOpportunityStatus)) {
                existingEntity.setOpportunityStatus(dtoIn.getOpportunityStatus());
                newOpportunityStatus = dtoIn.getOpportunityStatus();
                statusChanged = true;
            }
            if (dtoIn.getRateStatus() != null) {
                existingEntity.setRateStatus(dtoIn.getRateStatus());
            }
            if (dtoIn.getCompanyRateStatus() != null) {
                existingEntity.setCompanyRateStatus(dtoIn.getCompanyRateStatus());
            }
        }

        AppliedOpportunity saved = repository.save(getSelf().updateEntityUpdater(existingEntity));

        // Log status change if it occurred
        if (statusChanged) {
            eventPublisher.publishEvent(new OpportunityStatusChangedEvent(
                    this,
                    saved,
                    originalOpportunityStatus,
                    dtoIn.getOpportunityStatus(),
                    userRepository.findByFirebaseUserId(getCurrentUser(request)).orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "User")),
                    dtoIn.getNote()
            ));
            statusHistoryService.logStatusChange(
                    saved,
                    originalOpportunityStatus,
                    newOpportunityStatus,
                    "Status updated via full update (PUT)"
            );
        }

        return saved;
    }

    @Transactional
    public AppliedOpportunity updateOpportunityStatus(Long id, boolean accept) {
        String firebaseUid = permissionUtils.getUserId();

        log.info("GDPR: Operation=updateOpportunityStatus, FirebaseUID={}, OpportunityID={}, Action={}, Purpose=status_management",
                firebaseUid, id, accept ? "ACCEPT" : "REJECT");

        // Repository lookup bypasses the broad permission check baked into
        // findById — we apply role-aware checks below instead.
        AppliedOpportunity opportunity = repository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException(
                        "error.business.item_not_found", "Applied opportunity"));

        OpportunityStatus currentStatus = opportunity.getOpportunityStatus();
        OpportunityStatus newStatus = OpportunityStatus.getNextStatus(currentStatus, accept);

        if (!permissionUtils.isAdmin()) {
            assertCallerCanTransition(id, opportunity, currentStatus);
        }

        if (!currentStatus.canTransitionTo(newStatus)) {
            throw new BusinessRuleTranslatableException("error.business.invalid_state");
        }

        AppliedOpportunity saved = getSelf().updateOpportunityStatus(id, newStatus, opportunity.getNote());

        // GDPR: Log successful status update
        log.info("GDPR: Operation=updateOpportunityStatus_SUCCESS, FirebaseUID={}, OpportunityID={}, OldStatus={}, NewStatus={}",
                firebaseUid, saved.getId(), currentStatus, newStatus);

        // Log the status change
        String changeReason = String.format("Status %s via API (/status/update endpoint)",
                accept ? "accepted" : "rejected");
        statusHistoryService.logStatusChange(
                saved,
                currentStatus,
                newStatus,
                changeReason
        );

        return saved;
    }

    @Transactional
    public AppliedOpportunity updateOpportunityStatus(Long id, OpportunityStatus newStatus, String note) {
        AppliedOpportunity opportunity = findById(id);
        OpportunityStatus previousStatus = opportunity.getOpportunityStatus();

        // Update status
        opportunity.setOpportunityStatus(newStatus);
        AppliedOpportunity saved = repository.save(opportunity);

        // Publish event (will be handled AFTER_COMMIT by NotificationEventListener)
        eventPublisher.publishEvent(new OpportunityStatusChangedEvent(
                this,
                saved,
                previousStatus,
                newStatus,
                userRepository.findByFirebaseUserId(getCurrentUser(request)).orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "User")),
                note
        ));

        return saved;
    }


    @Override
    @Transactional
    public AppliedOpportunity patch(Long id, Map<String, Object> updates) {
        // Self-injection: route through the proxy so the @Transactional on
        // findById is honoured even though we are inside the same bean.
        AppliedOpportunity opportunity = getSelf().findById(id);

        assertPatchAllowed(id, opportunity, updates);

        OpportunityStatus originalStatus = opportunity.getOpportunityStatus();

        updateEnumField(updates, "rateStatus", RateStatus.class, opportunity::setRateStatus);
        updateEnumField(updates, "companyRateStatus", RateStatus.class, opportunity::setCompanyRateStatus);

        boolean statusChanged = false;
        if (permissionUtils.isAdmin() && updates.containsKey(OPPORTUNITY_STATUS_FIELD)) {
            updateEnumField(updates, OPPORTUNITY_STATUS_FIELD, OpportunityStatus.class,
                    opportunity::setOpportunityStatus);
            statusChanged = !originalStatus.equals(opportunity.getOpportunityStatus());
        }

        Set<String> ignoredFields = Set.of(
                "id", "createdTime", "lastUpdateTime",
                "rateStatus", "companyRateStatus", OPPORTUNITY_STATUS_FIELD);
        AppliedOpportunity saved = patch(id, updates, ignoredFields);

        if (statusChanged) {
            publishStatusChange(saved, originalStatus, "Status updated via patch operation (PATCH)");
        }
        return saved;
    }

    /**
     * Per-role authorization gate for {@link #patch(Long, Map)}.
     *
     * <p>Influencers may only patch when they identify themselves via the
     * {@code influencer} field on the payload AND that id matches their own
     * stored id. Companies must own the parent partnership opportunity.
     * Admins skip both checks.
     */
    private void assertPatchAllowed(Long id,
                                    AppliedOpportunity opportunity,
                                    Map<String, Object> updates) {
        if (permissionUtils.isAdmin()) {
            return;
        }

        if (permissionUtils.isInfluencer()) {
            if (!isPatchInfluencerOwner(updates)) {
                throw patchPermissionDenied(id);
            }
        }

        if (permissionUtils.isCompany() && !permissionUtils.isUserOwner(opportunity)) {
            throw patchPermissionDenied(id);
        }
    }

    private boolean isPatchInfluencerOwner(Map<String, Object> updates) {
        Object rawInfluencerId = updates.get(INFLUENCER_FIELD);
        if (!updates.containsKey(INFLUENCER_FIELD) || rawInfluencerId == null) {
            return false;
        }
        long claimedInfluencerId;
        try {
            claimedInfluencerId = Long.parseLong(rawInfluencerId.toString());
        } catch (NumberFormatException e) {
            throw new ValidationTranslatableException(
                    "error.validation.type_mismatch", "influencer ID");
        }
        return userRepository.findByFirebaseUserId(permissionUtils.getUserId())
                .map(user -> Long.valueOf(claimedInfluencerId).equals(user.getId()))
                .orElse(false);
    }

    private InsufficientPermissionsException patchPermissionDenied(Long id) {
        return new InsufficientPermissionsException(
                "error.auth.insufficient_permissions",
                permissionUtils.getUserId(),
                "patch",
                "AppliedOpportunity#" + id);
    }

    private void publishStatusChange(AppliedOpportunity saved,
                                     OpportunityStatus originalStatus,
                                     String changeReason) {
        eventPublisher.publishEvent(new OpportunityStatusChangedEvent(
                this,
                saved,
                originalStatus,
                saved.getOpportunityStatus(),
                userRepository.findByFirebaseUserId(getCurrentUser(request))
                        .orElseThrow(() -> new ResourceNotFoundException(
                                "error.business.item_not_found", "User")),
                saved.getNote()));
        statusHistoryService.logStatusChange(
                saved,
                originalStatus,
                saved.getOpportunityStatus(),
                changeReason);
    }

    /**
     * Updates the rating for an applied opportunity based on user role.
     * Influencers can rate companies, companies can rate influencers.
     *
     * @param id         the applied opportunity ID
     * @param ratingType "influencer" for influencer rating company, "company" for company rating influencer
     * @param rating     the rating value (POSITIVE, NEGATIVE, or DEFAULT)
     * @return the updated applied opportunity
     */
    public AppliedOpportunity updateRating(Long id, String ratingType, RateStatus rating) {
        String firebaseUid = permissionUtils.getUserId();

        // GDPR: Log rating update
        log.info("GDPR: Operation=updateRating, FirebaseUID={}, OpportunityID={}, RatingType={}, Rating={}, Purpose=feedback_management",
                firebaseUid, id, ratingType, rating);

        // Use repository directly to avoid restrictive permission check in findById
        AppliedOpportunity opportunity = repository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Applied opportunity"));

        // Permission checks based on rating type
        if (!permissionUtils.isAdmin()) {
            if (INFLUENCER_FIELD.equals(ratingType)) {
                // Influencer rating company - verify this influencer owns the application
                if (!permissionUtils.isInfluencer() ||
                        !opportunity.getInfluencer().getFirebaseUserId().equals(permissionUtils.getUserId())) {
                    throw new InsufficientPermissionsException(
                            "error.auth.insufficient_permissions",
                            firebaseUid,
                            "updateRating",
                            "AppliedOpportunity#" + id);
                }
                opportunity.setRateStatus(rating);
            } else if (COMPANY_FIELD.equals(ratingType)) {
                // Company rating influencer - verify this company owns the partnership opportunity
                if (!permissionUtils.isCompany() ||
                        !opportunity.getPartnershipOpportunity().getCompany().getFirebaseUserId().equals(permissionUtils.getUserId())) {
                    throw new InsufficientPermissionsException(
                            "error.auth.insufficient_permissions",
                            firebaseUid,
                            "updateRating",
                            "AppliedOpportunity#" + id);
                }
                opportunity.setCompanyRateStatus(rating);
            } else {
                throw new ValidationTranslatableException("error.validation.invalid_argument");
            }
        } else {
            // Admin can update any rating
            if (INFLUENCER_FIELD.equals(ratingType)) {
                opportunity.setRateStatus(rating);
            } else if (COMPANY_FIELD.equals(ratingType)) {
                opportunity.setCompanyRateStatus(rating);
            } else {
                throw new ValidationTranslatableException("error.validation.invalid_argument");
            }
        }

        AppliedOpportunity saved = repository.save(opportunity);

        // GDPR: Log successful rating update
        log.info("GDPR: Operation=updateRating_SUCCESS, FirebaseUID={}, OpportunityID={}, RatingType={}, NewRating={}",
                firebaseUid, saved.getId(), ratingType, rating);

        // Log the rating change
        String changeReason = String.format("%s rating updated to %s via API",
                ratingType.substring(0, 1).toUpperCase() + ratingType.substring(1), rating);
        statusHistoryService.logStatusChange(
                saved,
                saved.getOpportunityStatus(), // No status change, just rating
                saved.getOpportunityStatus(),
                changeReason
        );

        return saved;
    }

    private <T extends Enum<T>> void updateEnumField(Map<String, Object> updates,
                                                     String fieldName,
                                                     Class<T> enumClass,
                                                     Consumer<T> setter) {
        if (!updates.containsKey(fieldName)) {
            return;
        }
        try {
            String rawValue = updates.get(fieldName).toString().toUpperCase(Locale.ROOT);
            T enumValue = Enum.valueOf(enumClass, rawValue);
            setter.accept(enumValue);
        } catch (IllegalArgumentException e) {
            throw new ValidationTranslatableException("error.validation.invalid_argument");
        }
    }

    /**
     * Statuses an influencer is allowed to drive forward — the responses to
     * a company decision and the content-flow steps. Anything not in this
     * set must be left to admins or the company side.
     */
    private static final java.util.Set<OpportunityStatus> INFLUENCER_DRIVABLE_STATES =
            java.util.EnumSet.of(
                    OpportunityStatus.ACCEPTED_BY_COMPANY,
                    OpportunityStatus.ACCEPTED_BY_INFLUENCER,
                    OpportunityStatus.CONTENT_REJECTED,
                    OpportunityStatus.CONTENT_APPROVED,
                    OpportunityStatus.CONTENT_POSTED_REJECTED);

    /**
     * Statuses a company is NOT allowed to drive forward — the
     * influencer-side decision points. Reaching one of these means the
     * influencer needs to act next, not the company.
     */
    private static final java.util.Set<OpportunityStatus> COMPANY_LOCKED_STATES =
            java.util.EnumSet.of(
                    OpportunityStatus.ACCEPTED_BY_COMPANY,
                    OpportunityStatus.REJECTED_BY_COMPANY,
                    OpportunityStatus.ACCEPTED_BY_INFLUENCER);

    /**
     * Per-role gate for {@link #updateOpportunityStatus(Long, boolean)}.
     * Admins are caller-vetted upstream and skip this check entirely.
     *
     * <p>For an influencer caller: must own the opportunity, and the current
     * status must be one the influencer is allowed to drive forward.
     *
     * <p>For a company caller: must own the parent partnership opportunity,
     * and the current status must NOT be one of the influencer-only
     * decision points.
     */
    private void assertCallerCanTransition(Long id,
                                           AppliedOpportunity opportunity,
                                           OpportunityStatus currentStatus) {
        String callerUid = permissionUtils.getUserId();

        if (permissionUtils.isInfluencer()) {
            if (!opportunity.getInfluencer().getFirebaseUserId().equals(callerUid)) {
                throw permissionDenied(callerUid, id, null);
            }
            if (!INFLUENCER_DRIVABLE_STATES.contains(currentStatus)) {
                throw permissionDenied(callerUid, id, currentStatus);
            }
        }

        if (permissionUtils.isCompany()) {
            String partnershipOwnerUid = opportunity.getPartnershipOpportunity()
                    .getCompany()
                    .getFirebaseUserId();
            if (!partnershipOwnerUid.equals(callerUid)) {
                throw permissionDenied(callerUid, id, null);
            }
            if (COMPANY_LOCKED_STATES.contains(currentStatus)) {
                throw permissionDenied(callerUid, id, currentStatus);
            }
        }
    }

    private InsufficientPermissionsException permissionDenied(String callerUid,
                                                              Long opportunityId,
                                                              OpportunityStatus statusContext) {
        String resource = statusContext != null
                ? "AppliedOpportunity#" + opportunityId + " (status: " + statusContext + ")"
                : "AppliedOpportunity#" + opportunityId;
        return new InsufficientPermissionsException(
                "error.auth.insufficient_permissions",
                callerUid,
                "updateOpportunityStatus",
                resource);
    }

    @Override
    @Transactional
    public void delete(Long id) {
        String firebaseUid = permissionUtils.getUserId();

        // GDPR: Log opportunity deletion
        log.warn("GDPR: DELETION Operation=deleteAppliedOpportunity, FirebaseUID={}, OpportunityID={}, Purpose=opportunity_removal, LegalBasis=user_request",
                firebaseUid, id);

        // Use self-injection to call transactional method properly
        AppliedOpportunity appliedOpportunity = getSelf().findById(id);
        if (!permissionUtils.canEditAppliedOpportunity(appliedOpportunity)) {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    firebaseUid,
                    "delete",
                    "AppliedOpportunity#" + id);
        }
        if (!permissionUtils.isAdmin() && !appliedOpportunity.getOpportunityStatus().equals(OpportunityStatus.APPLIED)) {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    firebaseUid,
                    "delete",
                    "AppliedOpportunity#" + id + " (status not APPLIED)");
        }

        OpportunityStatus status = appliedOpportunity.getOpportunityStatus();
        repository.deleteById(id);

        // GDPR: Log successful deletion
        log.warn("GDPR: DELETION_COMPLETE Operation=deleteAppliedOpportunity_SUCCESS, FirebaseUID={}, OpportunityID={}, DeletedStatus={}, DataRemoved=application_data",
                firebaseUid, id, status);
    }

    public boolean existsByOpportunityIdAndStatusIn(Long opportunityId, Set<OpportunityStatus> statuses) {
        return ((AppliedOpportunityRepository) repository).existsByPartnershipOpportunity_IdAndOpportunityStatusIn(opportunityId, statuses);
    }

    public boolean existsByIdAndStatusIn(Long appliedOpportunityId, Set<OpportunityStatus> statuses) {
        return ((AppliedOpportunityRepository) repository).existsByIdAndOpportunityStatusIn(appliedOpportunityId, statuses);
    }

    public boolean isAppliedOpportunityActive(AppliedOpportunity ao) {
        return existsByIdAndStatusIn(ao.getId(), Set.of(
                OpportunityStatus.ACCEPTED_BY_COMPANY,
                OpportunityStatus.ACCEPTED_BY_INFLUENCER,
                OpportunityStatus.CONTENT_SEND_TO_ACCEPT,
                OpportunityStatus.CONTENT_APPROVED,
                OpportunityStatus.CONTENT_POSTED,
                OpportunityStatus.CONTENT_REJECTED,
                OpportunityStatus.CONTENT_POSTED_REJECTED
        ));
    }

    // Helper method to get user by Firebase ID
    public User getUserByFirebaseId(String firebaseUserId) {
        return userRepository.findByFirebaseUserId(firebaseUserId)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "User"));
    }

    /**
     * Get collaborations in progress with dynamic filtering based on user role.
     * Migrated from ActiveCooperationService.
     */
    @Transactional(readOnly = true)
    public MappingJacksonValue getCollaborationsInProgress(Pageable pageable, List<OpportunityStatus> statuses) {
        String firebaseUid = permissionUtils.getUserId();

        // GDPR: Log collaboration retrieval
        log.info("GDPR: Operation=getCollaborationsInProgress, FirebaseUID={}, Purpose=active_collaborations_monitoring, StatusesRequested={}",
                firebaseUid, statuses != null ? statuses.size() : "all");

        CoopFilter filter = new CoopFilter();
        List<OpportunityStatus> statusesOfPossibility = List.of(
                OpportunityStatus.ACCEPTED_BY_COMPANY,
                OpportunityStatus.ACCEPTED_BY_INFLUENCER,
                OpportunityStatus.CONTENT_SEND_TO_ACCEPT,
                OpportunityStatus.CONTENT_APPROVED,
                OpportunityStatus.CONTENT_REJECTED,
                OpportunityStatus.CONTENT_POSTED_REJECTED,
                OpportunityStatus.CONTENT_POSTED,
                OpportunityStatus.REJECTED_BY_COMPANY,
                OpportunityStatus.REJECTED_BY_INFLUENCER,
                OpportunityStatus.DONE
        );

        if (statuses != null && !statuses.isEmpty()) {
            // Check if provided statuses are valid
            if (!new HashSet<>(statusesOfPossibility).containsAll(statuses)) {
                throw new ValidationTranslatableException("error.validation.invalid_argument");
            }
            filter.setOpportunityStatuses(statuses);
        } else {
            filter.setOpportunityStatuses(statusesOfPossibility);
        }

        // Apply user-specific filtering
        // Check both role and user type for influencers - same as other filtering methods  
        if (permissionUtils.isInfluencer() || permissionUtils.isCurrentUserTypeInfluencer()) {
            String firebaseUserId = permissionUtils.getUserId();
            User influencer = userRepository.findByFirebaseUserId(firebaseUserId)
                    .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Influencer"));
            filter.setInfluencerId(influencer.getId());
        }

        filter.setFilterRateStatus(RateStatus.DEFAULT);
        applyCompanyFilter(filter);

        // Get filtered results
        Page<CoopDto> result = findAllCollaborations(pageable, filter);

        MappingJacksonValue mapping = new MappingJacksonValue(result.getContent());

        // Set serialization view based on user role
        if (permissionUtils.isAdmin()) {
            mapping.setSerializationView(Views.InProgress_AdminView.class);
        } else if (permissionUtils.isInfluencer()) {
            mapping.setSerializationView(Views.InProgress_InfluencerView.class);
        } else if (permissionUtils.isCompany()) {
            mapping.setSerializationView(Views.InProgress_CompanyView.class);
        }

        return mapping;
    }

    /**
     * Apply company-specific filtering if the current user is a company.
     * Migrated from ActiveCooperationService.
     */
    private void applyCompanyFilter(CoopFilter filter) {
        // Check both role and user type for companies - same as other filtering methods
        if (permissionUtils.isCompany() || permissionUtils.isCurrentUserTypeCompany()) {
            String firebaseUserId = permissionUtils.getUserId();
            User company = userRepository.findByFirebaseUserId(firebaseUserId)
                    .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Company"));
            filter.setCompanyId(company.getId());
        }
    }

    /**
     * Find all collaborations based on the provided filter.
     * Migrated from ActiveCooperationService.
     */
    private Page<CoopDto> findAllCollaborations(Pageable pageable, CoopFilter filter) {
        // Create specification based on filter
        Specification<AppliedOpportunity> spec = new AppliedOpportunitySpecification(filter);
        // Use the custom method with multiple queries to avoid MultipleBagFetchException
        Page<AppliedOpportunity> page = ((AppliedOpportunityRepository) repository).findAllWithAssociationsFetched(spec, pageable);

        // Map results to DTOs
        return page.map(CoopDto::mapToInfluencerCoopDto);
    }

    /**
     * Validates that the influencer's follower count is within the partnership opportunity's required range.
     *
     * @param influencer  The influencer applying to the opportunity
     * @param opportunity The partnership opportunity being applied to
     * @throws FollowerValidationException if follower count is not within required range
     */
    private void validateFollowerRange(User influencer, PartnershipOpportunity opportunity) {
        // Get primary social connection follower count
        Integer followerCount = getPrimaryFollowerCount(influencer.getId());

        // Check if follower data is available
        if (followerCount == null) {
            throw new FollowerValidationException(
                    "Unable to verify follower count. Please ensure your primary social media connection has up-to-date follower information."
            );
        }

        // Validate against opportunity requirements
        long minRequired = opportunity.getFollowersMin();
        long maxRequired = opportunity.getFollowersMax();

        if (followerCount < minRequired) {
            throw new FollowerValidationException(
                    String.format(Locale.US, "Your follower count (%,d) is below the minimum requirement (%,d) for this opportunity.",
                            followerCount, minRequired)
            );
        }

        // Only check maximum if it's greater than 0 (0 means unlimited)
        if (maxRequired > 0 && followerCount > maxRequired) {
            throw new FollowerValidationException(
                    String.format(Locale.US, "Your follower count (%,d) exceeds the maximum limit (%,d) for this opportunity.",
                            followerCount, maxRequired)
            );
        }
    }

    /**
     * Gets the follower count from the user's primary social media connection.
     *
     * @param userId The user ID to get follower count for
     * @return The follower count from primary connection, or null if not available
     */
    private Integer getPrimaryFollowerCount(Long userId) {
        return userSocialConnectionRepository.findByUserIdAndIsPrimaryTrue(userId)
                .filter(conn -> conn.getConnectionStatus() == ConnectionStatus.CONNECTED)
                .map(UserSocialConnection::getFollowersCount)
                .orElse(null);
    }

    /**
     * Public method to check if an influencer can apply to a partnership opportunity
     * based on follower count requirements. This can be used for pre-validation.
     *
     * @param influencerId  The influencer's user ID
     * @param opportunityId The partnership opportunity ID
     * @return FollowerValidationResult containing validation status and details
     */
    public FollowerValidationResult validateInfluencerEligibility(Long influencerId, Long opportunityId) {
        String firebaseUid = permissionUtils.getUserId();

        // GDPR: Log eligibility validation
        log.info("GDPR: Operation=validateInfluencerEligibility, FirebaseUID={}, InfluencerID={}, OpportunityID={}, Purpose=application_eligibility_check",
                firebaseUid, influencerId, opportunityId);

        User influencer = userRepository.findById(influencerId)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Influencer"));

        PartnershipOpportunity opportunity =
                (repositoryResolver
                        .getRepository(PartnershipOpportunity.class))
                        .findById(opportunityId)
                        .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Partnership opportunity"));

        try {
            validateFollowerRange(influencer, opportunity);
            return FollowerValidationResult.success(getPrimaryFollowerCount(influencerId));
        } catch (FollowerValidationException e) {
            return FollowerValidationResult.failure(e.getMessage(), getPrimaryFollowerCount(influencerId));
        }
    }

    /**
     * Updates company rating for an applied opportunity.
     * This method is specifically for companies to rate influencers and bypasses
     * the general edit permission check in the save method.
     *
     * @param id     the applied opportunity ID
     * @param rating the rating value (POSITIVE, NEGATIVE, or DEFAULT)
     * @return the updated applied opportunity
     */
    public AppliedOpportunity updateCompanyRating(Long id, RateStatus rating) {
        // Use repository directly to avoid restrictive permission check in findById
        AppliedOpportunity opportunity = repository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Applied opportunity"));

        // Permission check - verify this company owns the partnership opportunity
        String firebaseUid = permissionUtils.getUserId();
        if (!permissionUtils.isAdmin() && (!permissionUtils.isCompany() ||
                !opportunity.getPartnershipOpportunity().getCompany().getFirebaseUserId().equals(firebaseUid))) {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    firebaseUid,
                    "updateCompanyRating",
                    "AppliedOpportunity#" + id);
        }

        // Prevent re-rating - users can only rate once
        if (opportunity.getCompanyRateStatus() != null &&
                opportunity.getCompanyRateStatus() != RateStatus.DEFAULT) {
            throw new BusinessRuleTranslatableException("error.business.rating_already_submitted");
        }

        opportunity.setCompanyRateStatus(rating);
        opportunity.setUpdaterId(permissionUtils.getUserId());

        // Use repository.save directly to bypass the restrictive canEditAppliedOpportunity check
        AppliedOpportunity saved = repository.save(getSelf().updateEntityUpdater(opportunity));

        // Log the rating change
        String changeReason = String.format("Company rating updated to %s via API", rating);
        statusHistoryService.logStatusChange(
                saved,
                saved.getOpportunityStatus(), // No status change, just rating
                saved.getOpportunityStatus(),
                changeReason
        );

        return saved;
    }

    /**
     * Updates influencer rating for an applied opportunity.
     * This method is specifically for influencers to rate companies and bypasses
     * the general edit permission check in the save method.
     *
     * @param id     the applied opportunity ID
     * @param rating the rating value (POSITIVE, NEGATIVE, or DEFAULT)
     * @return the updated applied opportunity
     */
    public AppliedOpportunity updateInfluencerRating(Long id, RateStatus rating) {
        // Use repository directly to avoid restrictive permission check in findById
        AppliedOpportunity opportunity = repository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Applied opportunity"));

        // Permission check - verify this influencer owns the application
        String firebaseUid = permissionUtils.getUserId();
        if (!permissionUtils.isAdmin() && (!permissionUtils.isInfluencer() ||
                !opportunity.getInfluencer().getFirebaseUserId().equals(firebaseUid))) {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    firebaseUid,
                    "updateInfluencerRating",
                    "AppliedOpportunity#" + id);
        }

        // Prevent re-rating - users can only rate once
        if (opportunity.getRateStatus() != null &&
                opportunity.getRateStatus() != RateStatus.DEFAULT) {
            throw new BusinessRuleTranslatableException("error.business.rating_already_submitted");
        }

        opportunity.setRateStatus(rating);
        opportunity.setUpdaterId(permissionUtils.getUserId());

        // Use repository.save directly to bypass the restrictive canEditAppliedOpportunity check
        AppliedOpportunity saved = repository.save(getSelf().updateEntityUpdater(opportunity));

        // Log the rating change
        String changeReason = String.format("Influencer rating updated to %s via API", rating);
        statusHistoryService.logStatusChange(
                saved,
                saved.getOpportunityStatus(), // No status change, just rating
                saved.getOpportunityStatus(),
                changeReason
        );

        return saved;
    }

    /**
     * Retrieves a paginated list of applied opportunities as DTOs with all conversions done within transaction.
     * This prevents LazyInitializationException by ensuring all DTO mappings happen inside @Transactional.
     *
     * @param pageable Pagination parameters
     * @param filters  Filter parameters
     * @param <D>      The output DTO type
     * @return Page of DTOs with all lazy relationships properly loaded
     */
    @Override
    @Transactional(readOnly = true)
    public <D> Page<D> getDataPagedAndFilteredAsDtos(Pageable pageable, Map<String, String> filters) {
        // Get entities with all lazy relationships initialized using self-injection
        Page<AppliedOpportunity> page = getSelf().getDataPagedAndFiltered(pageable, filters);

        // Convert to DTOs within transaction boundary using the utility method
        Page<AppliedOpportunityDtoOut> dtoPage = page.map(this::mapToDto);

        // Cast to generic type
        @SuppressWarnings("unchecked")
        Page<D> result = (Page<D>) dtoPage;
        return result;
    }

    // ===== DTO CONVERSION METHODS =====

    /**
     * Converts an AppliedOpportunity entity to DTO.
     * Implements the abstract method from BaseService.
     *
     * @param entity The entity to convert
     * @return The converted DTO
     */
    @Override
    @Transactional(readOnly = true)
    public <D> D toDto(AppliedOpportunity entity) {
        @SuppressWarnings("unchecked")
        D result = (D) mapToDto(entity);
        return result;
    }

    /**
     * Creates a new entity from DTO and returns it as DTO.
     * Implements the abstract method from BaseService.
     *
     * @param dto The input DTO
     * @return The created entity as DTO
     */
    @Override
    @Transactional
    public <D> D createFromDtoAsDto(AppliedOpportunityDtoIn dto) {
        @SuppressWarnings("unchecked")
        // Use self-injection to call transactional method properly
        D result = (D) getSelf().saveAsDto(dto);
        return result;
    }

    /**
     * Converts a list of AppliedOpportunity entities to AppliedOpportunityDtoOut list.
     * Must be called within a transaction to avoid LazyInitializationException.
     *
     * @param entities The list of AppliedOpportunity entities to convert
     * @return List of AppliedOpportunityDtoOut
     */
    @Transactional(readOnly = true)
    public List<AppliedOpportunityDtoOut> toDtoList(List<AppliedOpportunity> entities) {
        if (entities == null || entities.isEmpty()) return new ArrayList<>();
        return entities.stream()
                .map(this::mapToDto)
                .toList();
    }

    // ===== FIND AS DTO METHODS =====

    /**
     * Finds an applied opportunity by ID and returns as DTO.
     * Combines fetch and conversion within transaction.
     *
     * @param id The applied opportunity ID
     * @return AppliedOpportunityDtoOut
     * @throws ItemNotFoundException if not found
     * @throws AccessDeniedException if user lacks permission
     */
    @Transactional(readOnly = true)
    @Override
    public AppliedOpportunityDtoOut findByIdAsDto(Long id) {
        // Use self-injection to call transactional method properly
        AppliedOpportunity entity = getSelf().findById(id);
        return mapToDto(entity);
    }

    /**
     * Creates a new applied opportunity and returns as DTO.
     *
     * @param dto The input DTO
     * @return AppliedOpportunityDtoOut of newly created entity
     */
    @Transactional
    public AppliedOpportunityDtoOut saveAsDto(AppliedOpportunityDtoIn dto) {
        // Handle influencer assignment before mapping
        String currentUserId = permissionUtils.getUserId();
        var currentUser = getUserByFirebaseId(currentUserId);
        if (!permissionUtils.isAdmin()) {
            // Three separate requirements used to be reported as one phrase, "missing requirements",
            // which named none of them. A caller then has to guess, and an e2e failure that is
            // really "the account is still IN_VALIDATION" looks exactly like "no social account
            // connected". The response body stays the same generic key -- an applicant learning
            // which precondition they fail is an enumeration aid -- but the server log says which.
            List<String> unmet = new ArrayList<>(3);
            if (currentUser.getSocialConnections().isEmpty()) {
                unmet.add("no social connection");
            }
            if (!currentUser.getAccountStatus().isActive()) {
                unmet.add("account status " + currentUser.getAccountStatus());
            }
            if (!permissionUtils.isInfluencer()) {
                unmet.add("role is not INFLUENCER");
            }
            if (!unmet.isEmpty()) {
                log.warn("GDPR: AccessDenied Operation=createAppliedOpportunity, FirebaseUID={}, Unmet={}",
                        currentUserId, unmet);
                throw new InsufficientPermissionsException(
                        "error.auth.insufficient_permissions",
                        currentUserId,
                        "saveAsDto",
                        "AppliedOpportunity (unmet: " + String.join("; ", unmet) + ")");
            }
        }
        if (dto.getInfluencer() == null) {
            if (permissionUtils.isInfluencer()) {
                // Get current user and set their ID in the DTO
                dto.setInfluencer(currentUser.getId());
            } else {
                // Non-influencer users must provide influencer ID
                throw new ValidationTranslatableException("error.validation.required_field", "influencer");
            }
        }

        AppliedOpportunity entity = modelMapper.map(dto, AppliedOpportunity.class);
        AppliedOpportunity savedEntity = save(entity);
        return mapToDto(savedEntity);
    }

    /**
     * Updates an applied opportunity and returns as DTO.
     *
     * @param id    The applied opportunity ID
     * @param dtoIn The input DTO with updates
     * @return AppliedOpportunityDtoOut of updated entity
     */
    @Transactional
    @Override
    public AppliedOpportunityDtoOut updateAsDto(Long id, AppliedOpportunityDtoIn dtoIn) {
        // Use self-injection to call transactional method properly
        AppliedOpportunity entity = getSelf().update(id, dtoIn);
        return mapToDto(entity);
    }

    /**
     * Patches an applied opportunity and returns as DTO.
     * Note: Renamed to avoid signature conflict with parent class.
     *
     * @param id      The applied opportunity ID
     * @param updates Map of field updates
     * @return AppliedOpportunityDtoOut of patched entity
     */
    @Transactional
    public AppliedOpportunityDtoOut patchEntityAsDto(Long id, Map<String, Object> updates) {
        // Use self-injection to call transactional method properly
        AppliedOpportunity entity = getSelf().patch(id, updates);
        return mapToDto(entity);
    }

    /**
     * Updates company rating and returns as DTO.
     *
     * @param id     The applied opportunity ID
     * @param rating The new rating
     * @return AppliedOpportunityDtoOut with updated rating
     */
    @Transactional
    public AppliedOpportunityDtoOut updateCompanyRatingAsDto(Long id, RateStatus rating) {
        AppliedOpportunity entity = updateCompanyRating(id, rating);
        return mapToDto(entity);
    }

    /**
     * Updates influencer rating and returns as DTO.
     *
     * @param id     The applied opportunity ID
     * @param rating The new rating
     * @return AppliedOpportunityDtoOut with updated rating
     */
    @Transactional
    public AppliedOpportunityDtoOut updateInfluencerRatingAsDto(Long id, RateStatus rating) {
        AppliedOpportunity entity = updateInfluencerRating(id, rating);
        return mapToDto(entity);
    }

    /**
     * Updates opportunity status and returns as DTO.
     *
     * @param id     The applied opportunity ID
     * @param accept Whether to accept or reject
     * @return AppliedOpportunityDtoOut with updated status
     */
    @Transactional
    public AppliedOpportunityDtoOut updateOpportunityStatusAsDto(Long id, boolean accept) {
        AppliedOpportunity entity = getSelf().updateOpportunityStatus(id, accept);
        return mapToDto(entity);
    }

    /**
     * Gets statistics for applied opportunities grouped by status categories for current user
     *
     * @return Statistics grouped by categories
     */
    @Transactional(readOnly = true)
    public AppliedOpportunityStatisticsDto getAppliedOpportunityStatistics() {
        String firebaseUserId = permissionUtils.getUserId();

        // GDPR: Log statistics retrieval
        log.info("GDPR: Operation=getAppliedOpportunityStatistics, FirebaseUID={}, Purpose=user_dashboard_statistics, DataAccessed=opportunity.counts",
                firebaseUserId);
        List<AppliedOpportunity> appliedOpportunities;

        AppliedOpportunityRepository appliedOpportunityRepository = (AppliedOpportunityRepository) repository;

        if (permissionUtils.isInfluencer() || permissionUtils.isCurrentUserTypeInfluencer()) {
            // Influencers see their own applied opportunities
            appliedOpportunities = appliedOpportunityRepository.findAllByInfluencer_FirebaseUserId(firebaseUserId);
        } else if (permissionUtils.isCompany() || permissionUtils.isCurrentUserTypeCompany()) {
            // Companies see applied opportunities to their partnership opportunities
            appliedOpportunities = appliedOpportunityRepository.findAllByPartnershipOpportunity_Company_FirebaseUserId(firebaseUserId);
        } else if (permissionUtils.isAdmin()) {
            appliedOpportunities = appliedOpportunityRepository.findAll();
        } else {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    firebaseUserId,
                    "getAppliedOpportunityStatistics",
                    "AppliedOpportunity statistics");
        }

        Map<String, Long> statusCounts = appliedOpportunities.stream()
                .collect(Collectors.groupingBy(
                        ao -> categorizeStatus(ao.getOpportunityStatus()),
                        Collectors.counting()
                ));

        return AppliedOpportunityStatisticsDto.builder()
                .inProgress(statusCounts.getOrDefault("inprogress", 0L))
                .newOpportunities(statusCounts.getOrDefault("new", 0L))
                .done(statusCounts.getOrDefault("done", 0L))
                .total((long) appliedOpportunities.size())
                .build();
    }

    /**
     * Categorizes opportunity status into one of three categories
     */
    private String categorizeStatus(OpportunityStatus status) {
        return switch (status) {
            case ACCEPTED_BY_INFLUENCER, CONTENT_SEND_TO_ACCEPT, CONTENT_APPROVED,
                 CONTENT_REJECTED, CONTENT_POSTED, CONTENT_POSTED_REJECTED, TO_BE_PAID -> "inprogress";
            case APPLIED, ACCEPTED_BY_COMPANY -> "new";
            case DONE, REJECTED_BY_COMPANY, REJECTED_BY_INFLUENCER -> "done";
        };
    }

    // ===== CIO-373: PAYMENT CONTACT METHODS =====

    /**
     * Get payment contact information for the other party in a collaboration.
     * Only available at TO_BE_PAID or DONE status.
     *
     * @param appliedOpportunityId The applied opportunity ID
     * @return PaymentContactDto with contact information
     * @throws ResourceNotFoundException        if the applied opportunity is not found
     * @throws InsufficientPermissionsException if user is not party to collaboration or status is not at payment stage
     */
    @Transactional(readOnly = true)
    public PaymentContactDto getPaymentContact(Long appliedOpportunityId) {
        String firebaseUid = permissionUtils.getUserId();

        // Fetch the applied opportunity
        AppliedOpportunity appliedOpportunity = repository.findById(appliedOpportunityId)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Applied Opportunity"));

        // Validate status is at payment stage
        OpportunityStatus status = appliedOpportunity.getOpportunityStatus();
        if (status != OpportunityStatus.TO_BE_PAID && status != OpportunityStatus.DONE) {
            log.warn("GDPR: ACCESS_DENIED getPaymentContact - Status {} not at payment stage, FirebaseUID={}, OpportunityID={}",
                    status, firebaseUid, appliedOpportunityId);
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    firebaseUid,
                    "getPaymentContact",
                    "Contact data only available at payment stage");
        }

        // Determine the "other party" based on current user
        User currentUser = userRepository.findByFirebaseUserId(firebaseUid)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "User"));

        User otherParty;
        boolean isCurrentUserInfluencer = appliedOpportunity.getInfluencer() != null &&
                appliedOpportunity.getInfluencer().getId().equals(currentUser.getId());
        boolean isCurrentUserCompanyOwner = appliedOpportunity.getPartnershipOpportunity() != null &&
                appliedOpportunity.getPartnershipOpportunity().getCompany() != null &&
                appliedOpportunity.getPartnershipOpportunity().getCompany().getId().equals(currentUser.getId());

        if (isCurrentUserInfluencer) {
            // Influencer requesting company contact
            otherParty = appliedOpportunity.getPartnershipOpportunity().getCompany();
        } else if (isCurrentUserCompanyOwner) {
            // Company requesting influencer contact
            otherParty = appliedOpportunity.getInfluencer();
        } else if (permissionUtils.isAdmin()) {
            // Admin can access - default to showing influencer contact
            otherParty = appliedOpportunity.getInfluencer();
        } else {
            log.warn("GDPR: ACCESS_DENIED getPaymentContact - User not party to collaboration, FirebaseUID={}, OpportunityID={}",
                    firebaseUid, appliedOpportunityId);
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    firebaseUid,
                    "getPaymentContact",
                    "User not party to this collaboration");
        }

        if (otherParty == null) {
            throw new ResourceNotFoundException("error.business.item_not_found", "Contact User");
        }

        // Check the other party's sharePhoneForPayments preference
        UserPreferences preferences = userPreferencesRepository.findByUserId(otherParty.getId());
        boolean sharePhone = preferences != null && Boolean.TRUE.equals(preferences.getSharePhoneForPayments());

        // Build and return the contact DTO
        String phone = sharePhone ? otherParty.getPhoneNumber() : null;

        log.info("GDPR: getPaymentContact_SUCCESS, FirebaseUID={}, OpportunityID={}, OtherPartyID={}, PhoneShared={}",
                firebaseUid, appliedOpportunityId, otherParty.getId(), sharePhone);

        return new PaymentContactDto(
                otherParty.getName(),
                otherParty.getEmail(),
                phone,
                otherParty.getProfilePicture()
        );
    }

    /**
     * Result class for follower validation checks
     */
    @Getter
    public static class FollowerValidationResult {
        private final boolean valid;
        private final String message;
        private final Integer currentFollowerCount;

        private FollowerValidationResult(boolean valid, String message, Integer currentFollowerCount) {
            this.valid = valid;
            this.message = message;
            this.currentFollowerCount = currentFollowerCount;
        }

        public static FollowerValidationResult success(Integer followerCount) {
            return new FollowerValidationResult(true, "Follower count meets requirements", followerCount);
        }

        public static FollowerValidationResult failure(String message, Integer followerCount) {
            return new FollowerValidationResult(false, message, followerCount);
        }

    }
}
