package com.sm.instagram.platform.partnershipopportunities;

import com.sm.instagram.platform.address.*;
import com.sm.instagram.platform.appliedopportunities.*;
import com.sm.instagram.platform.city.City;
import com.sm.instagram.platform.city.CityRepository;
import com.sm.instagram.platform.common.authorization.PermissionUtils;
import com.sm.instagram.platform.common.base.BaseRepository;
import com.sm.instagram.platform.common.base.BaseService;
import com.sm.instagram.platform.common.exceptions.AuthenticationTranslatableException;
import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.common.exceptions.InsufficientPermissionsException;
import com.sm.instagram.platform.common.exceptions.ResourceNotFoundException;
import com.sm.instagram.platform.common.exceptions.ValidationTranslatableException;
import com.sm.instagram.platform.common.translation.TranslationService;
import com.sm.instagram.platform.common.util.RepositoryResolver;
import com.sm.instagram.platform.common.util.filtering.SpecificationBuilder;
import com.sm.instagram.platform.contenttype.ContentType;
import com.sm.instagram.platform.contenttype.ContentTypeDtoOut;
import com.sm.instagram.platform.currency.Currency;
import com.sm.instagram.platform.currency.CurrencyDtoOut;
import com.sm.instagram.platform.dictionary.DictionaryService;
import com.sm.instagram.platform.platform.Platform;
import com.sm.instagram.platform.platform.PlatformDto;
import com.sm.instagram.platform.servicetype.ServiceType;
import com.sm.instagram.platform.servicetype.ServiceTypeDtoOut;
import com.sm.instagram.platform.user.CompanyPublicProfileDto;
import com.sm.instagram.platform.user.User;
import com.sm.instagram.platform.user.UserRepository;
import com.sm.instagram.platform.userpreferences.UserPreferences;
import com.sm.instagram.platform.userpreferences.UserPreferencesRepository;
import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.modelmapper.ModelMapper;
import org.springframework.context.ApplicationContext;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.domain.Specification;
import org.springframework.security.access.AccessDeniedException;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Propagation;
import org.springframework.transaction.annotation.Transactional;

import java.lang.reflect.Field;
import java.util.*;
import java.util.stream.Collectors;

@Service
@Slf4j
public class PartnershipOpportunityService extends BaseService<PartnershipOpportunity, Long, PartnershipOpportunityDtoIn> {


    public static final String ADDRESS = "address";
    public static final String ADDRESS_ID = "addressId";
    public static final String PHOTOS = "photos";
    public static final String ORDER_NUMBER = "orderNumber";
    public static final String IS_COVER = "isCover";
    public static final String ADDRESS_TYPE = "addressType";
    public static final String IS_PRIMARY = "isPrimary";
    // Constants to avoid SonarQube "Define a constant" warnings
    private static final String DEFAULT_ADDRESS_TYPE = "MAIN";
    private static final int MAX_PHOTOS = 6;
    private final PartnershipOpportunityRepository partnershipOpportunityRepository;
    private final AppliedOpportunityRepository appliedOpportunityRepository;
    private final AddressRepository addressRepository;
    private final AddressService addressService;
    private final PermissionUtils permissionUtils;
    private final TranslationService translationService;
    private final UserRepository userRepository;
    private final DictionaryService dictionaryService;
    private final UserPreferencesRepository userPreferencesRepository;
    private final HttpServletRequest request;
    private final com.sm.instagram.platform.subscription.CampaignLimitService campaignLimitService;
    private final com.sm.instagram.platform.storage.service.SignedUrlService signedUrlService;

    protected PartnershipOpportunityService(ApplicationContext applicationContext,
                                            SpecificationBuilder<PartnershipOpportunity> specificationBuilder,
                                            PartnershipOpportunityRepository repository,
                                            ModelMapper modelMapper,
                                            RepositoryResolver repositoryResolver,
                                            AppliedOpportunityRepository appliedOpportunityRepository,
                                            AddressRepository addressRepository,
                                            AddressService addressService,
                                            PermissionUtils permissionUtils,
                                            TranslationService translationService,
                                            UserRepository userRepository,
                                            DictionaryService dictionaryService,
                                            UserPreferencesRepository userPreferencesRepository,
                                            HttpServletRequest request,
                                            com.sm.instagram.platform.subscription.CampaignLimitService campaignLimitService,
                                            com.sm.instagram.platform.storage.service.SignedUrlService signedUrlService) {
        super(applicationContext, specificationBuilder, repository, modelMapper, repositoryResolver);
        this.partnershipOpportunityRepository = repository;
        this.appliedOpportunityRepository = appliedOpportunityRepository;
        this.addressRepository = addressRepository;
        this.addressService = addressService;
        this.permissionUtils = permissionUtils;
        this.translationService = translationService;
        this.userRepository = userRepository;
        this.dictionaryService = dictionaryService;
        this.userPreferencesRepository = userPreferencesRepository;
        this.request = request;
        this.campaignLimitService = campaignLimitService;
        this.signedUrlService = signedUrlService;
    }

    /**
     * Resolves a NEW photo's tracked upload to its BE-minted URL (pentest 3.1
     * + owner directive 2026-06-13: the client never supplies a photo URL —
     * the server derives it from its own {@code file_uploads} row after
     * verifying ownership and that the blob landed in our bucket).
     */
    private String resolveNewPhotoUrl(String uploadId) {
        if (uploadId == null || uploadId.isBlank()) {
            throw new ValidationTranslatableException("validation.photo.uploadId.required");
        }
        return signedUrlService.resolveOwnedUpload(permissionUtils.getUserId(), uploadId).publicUrl();
    }

    /**
     * Gets the proxied instance of this service to ensure @Transactional methods work correctly.
     * This avoids circular dependency issues while ensuring proper transaction management.
     */
    @Override
    protected PartnershipOpportunityService getSelf() {
        return (PartnershipOpportunityService) super.getSelf();
    }

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
     * Converts a PartnershipOpportunity entity to DTO.
     * Implements the abstract method from BaseService.
     *
     * @param entity the entity to convert
     * @return the converted DTO
     */
    @Override
    @Transactional(readOnly = true)
    public <DTOOUT> DTOOUT toDto(PartnershipOpportunity entity) {
        @SuppressWarnings("unchecked")
        DTOOUT result = (DTOOUT) getSelf().convertToFilteredDto(entity);
        return result;
    }

    /**
     * Creates a new entity from DTO and returns it as DTO.
     * Implements the abstract method from BaseService.
     *
     * @param dto the input DTO
     * @return the created entity as DTO
     */
    @Override
    @Transactional
    public <DTOOUT> DTOOUT createFromDtoAsDto(PartnershipOpportunityDtoIn dto) {
        @SuppressWarnings("unchecked")
        DTOOUT result = (DTOOUT) getSelf().saveFromDtoAsDto(dto);
        return result;
    }

    /**
     * Converts a PartnershipOpportunity entity to DTO with properly filtered applied opportunities.
     *
     * @param entity the partnership opportunity entity to convert
     * @return DTO with filtered applied opportunities based on user role
     */
    @Transactional(readOnly = true)
    public PartnershipOpportunityDtoOut convertToFilteredDto(PartnershipOpportunity entity) {
        return getSelf().convertToFilteredDto(entity, getLocaleFromRequest()); // Use request locale instead of default
    }

    /**
     * Converts a PartnershipOpportunity entity to DTO with properly filtered applied opportunities and translations.
     *
     * @param entity the partnership opportunity entity to convert
     * @param locale the locale for translations
     * @return DTO with filtered applied opportunities based on user role and translated fields
     */
    @Transactional(readOnly = true)
    public PartnershipOpportunityDtoOut convertToFilteredDto(PartnershipOpportunity entity, Locale locale) {
        // Manual mapping to avoid ModelMapper configuration issues
        PartnershipOpportunityDtoOut dto = new PartnershipOpportunityDtoOut();

        // Map basic fields
        dto.setId(entity.getId());
        dto.setName(entity.getName());
        dto.setTitle(entity.getTitle());
        dto.setDetails(entity.getDetails());
        dto.setRequirements(entity.getRequirements());
        dto.setFollowersMin(entity.getFollowersMin());
        dto.setFollowersMax(entity.getFollowersMax());
        // Map CompensationType with translation
        if (entity.getCompensationType() != null) {
            CompensationTypeDtoOut compensationTypeDto = CompensationTypeDtoOut.builder()
                    .value(entity.getCompensationType().name())
                    .label(entity.getCompensationType().getLabel(dictionaryService, locale))
                    .originalLabel(entity.getCompensationType().name())
                    .build();
            dto.setCompensationType(compensationTypeDto);
        }
        // Coalesce nullable compensation (legacy rows) to 0 so the int-typed DtoOut contract is preserved.
        dto.setCompensationAmountMin(entity.getCompensationAmountMin() != null ? entity.getCompensationAmountMin() : 0);
        dto.setCompensationAmountMax(entity.getCompensationAmountMax() != null ? entity.getCompensationAmountMax() : 0);
        dto.setCompensationDescription(entity.getCompensationDescription());
        dto.setStartDate(entity.getStartDate());
        dto.setEndDate(entity.getEndDate());
        dto.setActive(entity.isActive());
        dto.setCreatedTime(entity.getCreatedTime());
        dto.setLastUpdateTime(entity.getLastUpdateTime());
        dto.setVersion(entity.getVersion());
        // Map City name
        if (entity.getCity() != null) {
            dto.setCity(entity.getCity().getName());
        }

        // Map Company
        if (entity.getCompany() != null) {
            CompanyPublicProfileDto companyDto = modelMapper.map(entity.getCompany(), CompanyPublicProfileDto.class);

            // Handle AccountStatus translation manually for company
            if (entity.getCompany().getAccountStatus() != null) {
                com.sm.instagram.platform.user.AccountStatusDtoOut accountStatusDtoOut = com.sm.instagram.platform.user.AccountStatusDtoOut.builder()
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

            // Always expose company email
            companyDto.setEmail(entity.getCompany().getEmail());

            // Expose phone only if company's sharePhoneForPayments preference allows it
            UserPreferences companyPrefs = userPreferencesRepository.findByUserId(entity.getCompany().getId());
            if (companyPrefs != null && Boolean.TRUE.equals(companyPrefs.getSharePhoneForPayments())) {
                companyDto.setPhoneNumber(entity.getCompany().getPhoneNumber());
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

        // Now filter and set the applied opportunities based on user role
        List<com.sm.instagram.platform.appliedopportunities.AppliedOpportunitySimpleDtoOut> filteredAppliedOpportunities =
                filterAppliedOpportunitiesForUser(entity.getAppliedOpportunities());
        dto.setAppliedOpportunities(filteredAppliedOpportunities);

        // Add translations for ServiceType
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

        // Add translations for Currency
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

        // Add translations for ContentTypes
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

        return dto;
    }

    /**
     * Filters applied opportunities based on the current user's role and permissions.
     *
     * @param appliedOpportunities the full list of applied opportunities to filter
     * @return filtered list based on user permissions:
     * - Admins: see all applied opportunities
     * - Companies: see applied opportunities ONLY for their own partnership opportunities
     * - Influencers: see only their own applications
     */
    private List<AppliedOpportunitySimpleDtoOut> filterAppliedOpportunitiesForUser(
            List<AppliedOpportunity> appliedOpportunities) {

        if (appliedOpportunities == null || appliedOpportunities.isEmpty()) {
            return new ArrayList<>();
        }

        if (permissionUtils.isAdmin()) {
            return appliedOpportunities.stream()
                    .map(this::convertAppliedOpportunityToDto)
                    .toList();
        }

        if (permissionUtils.isCompany() || permissionUtils.isCurrentUserTypeCompany()) {
            String currentUserId = permissionUtils.getUserId();
            return appliedOpportunities.stream()
                    .filter(ao -> ao.getPartnershipOpportunity() != null &&
                            ao.getPartnershipOpportunity().getCompany() != null &&
                            currentUserId.equals(ao.getPartnershipOpportunity().getCompany().getFirebaseUserId()))
                    .map(this::convertAppliedOpportunityToDto)
                    .toList();
        }

        if (permissionUtils.isInfluencer() || permissionUtils.isCurrentUserTypeInfluencer()) {
            String currentUserId = permissionUtils.getUserId();
            return appliedOpportunities.stream()
                    .filter(ao -> ao.getInfluencer() != null &&
                            currentUserId.equals(ao.getInfluencer().getFirebaseUserId()))
                    .map(this::convertAppliedOpportunityToDto)
                    .toList();
        }

        return new ArrayList<>();
    }

    /**
     * Converts an AppliedOpportunity entity to AppliedOpportunitySimpleDtoOut,
     * ensuring that the influencer field is properly converted to a DTO.
     *
     * @param appliedOpportunity the entity to convert
     * @return the converted DTO with properly mapped influencer
     */
    private AppliedOpportunitySimpleDtoOut convertAppliedOpportunityToDto(AppliedOpportunity appliedOpportunity) {
        AppliedOpportunitySimpleDtoOut dto = modelMapper.map(appliedOpportunity, AppliedOpportunitySimpleDtoOut.class);

        Locale locale = getLocaleFromRequest();

        // Handle OpportunityStatus translation manually
        if (appliedOpportunity.getOpportunityStatus() != null) {
            OpportunityStatusDtoOut opportunityStatusDtoOut = OpportunityStatusDtoOut.builder()
                    .value(appliedOpportunity.getOpportunityStatus().name())
                    .label(appliedOpportunity.getOpportunityStatus().getLabel(dictionaryService, locale))
                    .description(appliedOpportunity.getOpportunityStatus().getDescription(dictionaryService, locale))
                    .originalLabel(appliedOpportunity.getOpportunityStatus().name())
                    .colorTheme(appliedOpportunity.getOpportunityStatus().getColorTheme())
                    .icon(appliedOpportunity.getOpportunityStatus().getIcon())
                    .aliases(appliedOpportunity.getOpportunityStatus().getAliases())
                    .possibleTransitions(appliedOpportunity.getOpportunityStatus().getPossibleTransitions().stream()
                            .map(Enum::name).toList())
                    .isTerminal(appliedOpportunity.getOpportunityStatus().isTerminalStatus())
                    .isSuccessful(appliedOpportunity.getOpportunityStatus().isSuccessfulCompletion())
                    .build();
            dto.setOpportunityStatus(opportunityStatusDtoOut);
        }

        // Handle RateStatus translation manually based on user role
        if (permissionUtils.isAdmin()) {
            // Admin sees both ratings
            if (appliedOpportunity.getRateStatus() != null) {
                RateStatusDtoOut rateStatusDtoOut = RateStatusDtoOut.builder()
                        .value(appliedOpportunity.getRateStatus().name())
                        .label(appliedOpportunity.getRateStatus().getLabel(dictionaryService, locale))
                        .originalLabel(appliedOpportunity.getRateStatus().name())
                        .colorTheme(appliedOpportunity.getRateStatus().getColorTheme())
                        .icon(appliedOpportunity.getRateStatus().getIcon())
                        .build();
                dto.setRateStatus(rateStatusDtoOut);
            }
            if (appliedOpportunity.getCompanyRateStatus() != null) {
                RateStatusDtoOut companyRateStatusDtoOut = RateStatusDtoOut.builder()
                        .value(appliedOpportunity.getCompanyRateStatus().name())
                        .label(appliedOpportunity.getCompanyRateStatus().getLabel(dictionaryService, locale))
                        .originalLabel(appliedOpportunity.getCompanyRateStatus().name())
                        .colorTheme(appliedOpportunity.getCompanyRateStatus().getColorTheme())
                        .icon(appliedOpportunity.getCompanyRateStatus().getIcon())
                        .build();
                dto.setCompanyRateStatus(companyRateStatusDtoOut);
            }
        } else if (permissionUtils.isInfluencer()) {
            // Influencer sees only the rating they received from the company (companyRateStatus)
            dto.setRateStatus(null); // Hide influencer's own rating of the company
            if (appliedOpportunity.getCompanyRateStatus() != null) {
                RateStatusDtoOut companyRateStatusDtoOut = RateStatusDtoOut.builder()
                        .value(appliedOpportunity.getCompanyRateStatus().name())
                        .label(appliedOpportunity.getCompanyRateStatus().getLabel(dictionaryService, locale))
                        .originalLabel(appliedOpportunity.getCompanyRateStatus().name())
                        .colorTheme(appliedOpportunity.getCompanyRateStatus().getColorTheme())
                        .icon(appliedOpportunity.getCompanyRateStatus().getIcon())
                        .build();
                dto.setCompanyRateStatus(companyRateStatusDtoOut);
            }
        } else if (permissionUtils.isCompany()) {
            // Company sees only the rating they received from the influencer (rateStatus)
            dto.setCompanyRateStatus(null); // Hide company's own rating of the influencer
            if (appliedOpportunity.getRateStatus() != null) {
                RateStatusDtoOut rateStatusDtoOut = RateStatusDtoOut.builder()
                        .value(appliedOpportunity.getRateStatus().name())
                        .label(appliedOpportunity.getRateStatus().getLabel(dictionaryService, locale))
                        .originalLabel(appliedOpportunity.getRateStatus().name())
                        .colorTheme(appliedOpportunity.getRateStatus().getColorTheme())
                        .icon(appliedOpportunity.getRateStatus().getIcon())
                        .build();
                dto.setRateStatus(rateStatusDtoOut);
            }
        } else {
            // For any other user type, clear both rate statuses for security
            dto.setRateStatus(null);
            dto.setCompanyRateStatus(null);
        }

        // Ensure influencer is converted to DTO, not left as entity
        if (appliedOpportunity.getInfluencer() != null) {
            com.sm.instagram.platform.user.InfluencerPublicProfileDto influencerDto = modelMapper.map(
                    appliedOpportunity.getInfluencer(),
                    com.sm.instagram.platform.user.InfluencerPublicProfileDto.class
            );

            // Handle AccountStatus translation manually for influencer
            if (appliedOpportunity.getInfluencer().getAccountStatus() != null) {
                com.sm.instagram.platform.user.AccountStatusDtoOut accountStatusDtoOut = com.sm.instagram.platform.user.AccountStatusDtoOut.builder()
                        .value(appliedOpportunity.getInfluencer().getAccountStatus().name())
                        .label(appliedOpportunity.getInfluencer().getAccountStatus().getLabel(dictionaryService, locale))
                        .description(appliedOpportunity.getInfluencer().getAccountStatus().getDescription(dictionaryService, locale))
                        .originalLabel(appliedOpportunity.getInfluencer().getAccountStatus().name())
                        .colorTheme(appliedOpportunity.getInfluencer().getAccountStatus().getColorTheme())
                        .icon(appliedOpportunity.getInfluencer().getAccountStatus().getIcon())
                        .isActive(appliedOpportunity.getInfluencer().getAccountStatus().isActive())
                        .canLogin(appliedOpportunity.getInfluencer().getAccountStatus().canLogin())
                        .isTerminal(appliedOpportunity.getInfluencer().getAccountStatus().isTerminal())
                        .build();
                influencerDto.setAccountStatus(accountStatusDtoOut);
            }

            dto.setInfluencer(influencerDto);
        }

        return dto;
    }

    /**
     * Retrieves all partnership opportunities associated with a specific company.
     *
     * @param company the company (user) whose opportunities should be fetched
     * @return a list of partnership opportunities for the given company
     */
    public List<PartnershipOpportunity> findByCompany(User company) {
        return partnershipOpportunityRepository.findByCompany(company);
    }

    /**
     * Finds a partnership opportunity by its ID and checks view permissions.
     *
     * @param id the ID of the opportunity to retrieve
     * @return the found partnership opportunity
     * @throws ResourceNotFoundException if the opportunity is not found
     * @throws AccessDeniedException     if the current user cannot view the opportunity
     */
    @Override
    public PartnershipOpportunity findById(Long id) {
        String userId = permissionUtils.getUserId();
        log.info("GDPR: Service=findOpportunityById, Operation=RETRIEVE_OPPORTUNITY, UserID={}, OpportunityID={}, Purpose=data_retrieval",
                userId, id);

        PartnershipOpportunity opportunity = repository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("error.business.item_not_found", "Opportunity"));

        if (!permissionUtils.canViewOpportunity(opportunity)) {
            log.warn("GDPR: Service=findOpportunityById, Operation=ACCESS_DENIED, UserID={}, OpportunityID={}, Purpose=unauthorized_access_attempt",
                    userId, id);
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    userId,
                    "findById",
                    "PartnershipOpportunity#" + id);
        }

        return opportunity;
    }

    /**
     * Retrieves all active partnership opportunities.
     *
     * @return a list of active partnership opportunities
     */
    private List<PartnershipOpportunity> findActiveOpportunities() {
        return partnershipOpportunityRepository.findByActive(true);
    }

    /**
     * Retrieves partnership opportunities based on their active status.
     *
     * @param active whether to retrieve active or inactive opportunities
     * @return a list of partnership opportunities matching the status
     */
    public List<PartnershipOpportunity> findOpportunitiesByStatus(boolean active) {
        return partnershipOpportunityRepository.findByActive(active);
    }

    /**
     * Retrieves a paginated and filtered list of partnership opportunities as DTOs.
     * This method ensures all DTO conversions happen within the transaction boundary.
     * <p>
     * Applies role-based restrictions:
     * - Admins: see all opportunities
     * - Companies: see only their own opportunities (consistent with AppliedOpportunityService)
     * - Influencers: see active opportunities for browsing/applying
     *
     * @param pageable the paging and sorting information
     * @param filters  the map of filters to apply
     * @param locale   the locale for translations
     * @return a page of filtered partnership opportunity DTOs
     */
    @Transactional(readOnly = true)
    public Page<PartnershipOpportunityDtoOut> getDataPagedAndFilteredAsDtos(Pageable pageable, Map<String, String> filters, Locale locale) {
        Page<PartnershipOpportunity> entityPage = getDataPagedAndFiltered(pageable, filters);
        // Convert to DTOs within the transaction
        return entityPage.map(entity -> getSelf().convertToFilteredDto(entity, locale));
    }

    /**
     * Retrieves a paginated list of entities as DTOs with all conversions done within transaction.
     * This prevents LazyInitializationException by ensuring all DTO mappings happen inside @Transactional.
     * This is the generic method required by BaseService.
     *
     * @param pageable Pagination parameters
     * @param filters  Filter parameters
     * @param <DTOOUT> The output DTO type
     * @return Page of DTOs with all lazy relationships properly loaded
     */
    @Override
    public <DTOOUT> Page<DTOOUT> getDataPagedAndFilteredAsDtos(Pageable pageable, Map<String, String> filters) {
        // Use request locale instead of default
        Page<PartnershipOpportunityDtoOut> dtoPage = getSelf().getDataPagedAndFilteredAsDtos(pageable, filters, getLocaleFromRequest());

        // Cast to generic type
        @SuppressWarnings("unchecked")
        Page<DTOOUT> result = (Page<DTOOUT>) dtoPage;
        return result;
    }

    /**
     * Finds opportunity by ID and returns as DTO with properly loaded relationships.
     * This method ensures all DTO conversions happen within the same transaction.
     *
     * @param id the ID of the opportunity
     * @return the opportunity as DTO
     */
    @Override
    @Transactional(readOnly = true)
    public PartnershipOpportunityDtoOut findByIdAsDto(Long id) {
        PartnershipOpportunity entity = findById(id);
        return getSelf().convertToFilteredDto(entity, getLocaleFromRequest());
    }

    /**
     * Finds opportunity by ID and returns as DTO with translations.
     * This method ensures all DTO conversions happen within the same transaction.
     *
     * @param id     the ID of the opportunity
     * @param locale the locale for translations
     * @return the opportunity as DTO with translations
     */
    @Transactional(readOnly = true)
    public PartnershipOpportunityDtoOut findByIdAsDto(Long id, Locale locale) {
        PartnershipOpportunity entity = findById(id);
        return getSelf().convertToFilteredDto(entity, locale);
    }

    /**
     * Creates a new opportunity and returns as DTO.
     * This method ensures all DTO conversions happen within the same transaction.
     *
     * @param dto the input DTO
     * @return the created opportunity as DTO
     */
    @Transactional
    public PartnershipOpportunityDtoOut saveFromDtoAsDto(PartnershipOpportunityDtoIn dto) {
        PartnershipOpportunity entity = getSelf().saveFromDto(dto);
        return getSelf().convertToFilteredDto(entity, getLocaleFromRequest());
    }

    /**
     * Creates a new opportunity and returns as DTO with translations.
     * This method ensures all DTO conversions happen within the same transaction.
     *
     * @param dto    the input DTO
     * @param locale the locale for translations
     * @return the created opportunity as DTO with translations
     */
    @Transactional
    public PartnershipOpportunityDtoOut saveFromDtoAsDto(PartnershipOpportunityDtoIn dto, Locale locale) {
        PartnershipOpportunity entity = getSelf().saveFromDto(dto);
        return getSelf().convertToFilteredDto(entity, locale);
    }

    /**
     * Updates an opportunity and returns as DTO.
     * This method ensures all DTO conversions happen within the same transaction.
     *
     * @param id  the ID of the opportunity
     * @param dto the input DTO
     * @return the updated opportunity as DTO
     */
    @Override
    @Transactional
    public PartnershipOpportunityDtoOut updateAsDto(Long id, PartnershipOpportunityDtoIn dto) {
        PartnershipOpportunity entity = getSelf().update(id, dto);
        return getSelf().convertToFilteredDto(entity, getLocaleFromRequest());
    }

    /**
     * Updates an opportunity and returns as DTO with translations.
     * This method ensures all DTO conversions happen within the same transaction.
     *
     * @param id     the ID of the opportunity
     * @param dto    the input DTO
     * @param locale the locale for translations
     * @return the updated opportunity as DTO with translations
     */
    @Transactional
    public PartnershipOpportunityDtoOut updateAsDto(Long id, PartnershipOpportunityDtoIn dto, Locale locale) {
        PartnershipOpportunity entity = getSelf().update(id, dto);
        return getSelf().convertToFilteredDto(entity, locale);
    }

    /**
     * Patches an opportunity and returns as DTO with translations.
     * This method ensures all DTO conversions happen within the same transaction.
     *
     * @param id      the ID of the opportunity
     * @param updates the map of updates
     * @param locale  the locale for translations
     * @return the patched opportunity as DTO with translations
     */
    @Transactional
    public PartnershipOpportunityDtoOut patchAsDto(Long id, Map<String, Object> updates, Locale locale) {
        PartnershipOpportunity entity = getSelf().patch(id, updates);
        return getSelf().convertToFilteredDto(entity, locale);
    }

    /**
     * Retrieves a paginated and filtered list of partnership opportunities.
     * <p>
     * Applies role-based restrictions:
     * - Admins: see all opportunities
     * - Companies: see ALL opportunities (can edit/create only their own, but can view all)
     * - Influencers: see active opportunities for browsing/applying (checks both role and user type)
     *
     * @param pageable the paging and sorting information
     * @param filters  the map of filters to apply
     * @return a page of filtered and accessible partnership opportunities
     */
    @Override
    public Page<PartnershipOpportunity> getDataPagedAndFiltered(Pageable pageable, Map<String, String> filters) {
        Specification<PartnershipOpportunity> spec = createSpecification(filters);

        if (!permissionUtils.isAdmin()) {
            // Companies can see ALL opportunities - no additional filtering needed
            if (permissionUtils.isCompany() || permissionUtils.isCurrentUserTypeCompany()) {
                // Companies can see ALL opportunities (no additional filtering)
                // Edit/create restrictions are handled in individual methods
            }
            // Influencers see only active opportunities (for browsing/applying)
            else if (permissionUtils.isInfluencer() || permissionUtils.isCurrentUserTypeInfluencer()) {
                spec = spec.and((root, query, criteriaBuilder) ->
                        criteriaBuilder.isTrue(root.get("active"))
                );
            } else {
                // Unknown user type - show nothing for security
                spec = spec.and((root, query, criteriaBuilder) ->
                        criteriaBuilder.disjunction() // Always false
                );
            }
        }

        return repository.findAll(spec, pageable);
    }

    /**
     * Saves a new partnership opportunity from the given DTO.
     *
     * @param dto the data transfer object containing opportunity details
     * @return the saved partnership opportunity entity
     */
    @Transactional(propagation = Propagation.REQUIRED)
    public PartnershipOpportunity saveFromDto(PartnershipOpportunityDtoIn dto) {
        String userId = permissionUtils.getUserId();
        log.info("GDPR: Service=saveOpportunity, Operation=CREATE_OPPORTUNITY, UserID={}, CompanyID={}, Purpose=business_operation",
                userId, dto.getCompany());

        // A partnership opportunity always belongs to a company (company_id is
        // NOT NULL on the entity). Fail fast with a clear business error rather
        // than letting a null company slip through to a DB constraint violation
        // at flush/commit — which, in a @Transactional test, surfaces only after
        // the assertion (admins otherwise skip the ownership check below).
        if (dto.getCompany() == null) {
            throw new ResourceNotFoundException("error.business.item_not_found", "Company");
        }

        // Validate that non-admin users can only create opportunities for their own company
        if (!permissionUtils.isAdmin() && dto.getCompany() != null) {
            try {
                BaseRepository<User, Long> userRepo =
                        (BaseRepository<User, Long>) repositoryResolver.getRepository(User.class);
                Optional<User> company = userRepo.findById(dto.getCompany());

                if (company.isPresent()) {
                    String currentUserId = permissionUtils.getUserId();
                    String companyUserId = company.get().getFirebaseUserId();

                    if (!companyUserId.equals(currentUserId)) {
                        throw new InsufficientPermissionsException(
                                "error.auth.insufficient_permissions",
                                currentUserId,
                                "saveFromDto",
                                "PartnershipOpportunity");
                    }
                    if (!permissionUtils.isCompany()) {
                        throw new InsufficientPermissionsException(
                                "error.auth.insufficient_permissions",
                                currentUserId,
                                "saveFromDto",
                                "PartnershipOpportunity");
                    }
                } else {
                    throw new ResourceNotFoundException("error.business.item_not_found", dto.getCompany());
                }
            } catch (InsufficientPermissionsException | AuthenticationTranslatableException | ResourceNotFoundException e) {
                throw e; // Re-throw these specific exceptions
            } catch (Exception e) {
                log.error("Error validating company ownership for opportunity creation: {}", e.getMessage());
                throw new InsufficientPermissionsException(
                        "error.auth.insufficient_permissions",
                        permissionUtils.getUserId(),
                        "saveFromDto",
                        "PartnershipOpportunity");
            }
        }

        // Enforce campaign limit for company users
        if (permissionUtils.isCompany() && dto.getCompany() != null) {
            campaignLimitService.enforceLimit(dto.getCompany());
        }

        PartnershipOpportunity entity = modelMapper.map(dto, PartnershipOpportunity.class);

        // Process address
        processAddress(entity, dto);

        // Process photos — a NEW campaign has only new photos, so every entry
        // must carry a tracked uploadId; the URL is BE-derived.
        if (dto.getPhotos() != null && !dto.getPhotos().isEmpty()) {
            List<PartnershipOpportunityPhoto> photos = new ArrayList<>();
            for (PartnershipOpportunityPhotoDtoIn photoDto : dto.getPhotos()) {
                PartnershipOpportunityPhoto photo = new PartnershipOpportunityPhoto();
                photo.setUrl(resolveNewPhotoUrl(photoDto.getUploadId()));
                photo.setOrderNumber(photoDto.getOrderNumber());
                photo.setIsCover(photoDto.getIsCover());
                photo.setPartnershipOpportunity(entity);
                photos.add(photo);
            }
            entity.setPhotos(photos);
        }

        return repository.save(entity);
    }

    /**
     * Process the address information from the DTO and set it on the PartnershipOpportunity entity.
     * Either creates a new address or uses an existing one based on the DTO content.
     *
     * @param entity The PartnershipOpportunity entity to update
     * @param dto    The DTO containing address information
     */
    private void processAddress(PartnershipOpportunity entity, PartnershipOpportunityDtoIn dto) {
        log.debug("Processing address for partnership opportunity");

        // Case 1: Using an existing address by ID
        if (dto.getAddressId() != null) {
            log.debug("Using existing address with ID: {}", dto.getAddressId());

            Address address = addressService.resolveAddressForNewOpportunity(dto.getAddressId());

            // Set up the bidirectional relationship properly
            entity.setAddress(address);

            // Manually manage the inverse relationship for validation
            if (address.getPartnershipOpportunities() == null) {
                address.setPartnershipOpportunities(new ArrayList<>());
            }
            if (!address.getPartnershipOpportunities().contains(entity)) {
                address.getPartnershipOpportunities().add(entity);
            }
        }
        // Case 2: Creating a new address from the provided data
        else if (dto.getAddress() != null) {
            log.debug("Creating new address from DTO data");

            // If entity already has an address, update it instead of creating a new one
            if (entity.getAddress() != null && entity.getId() != null) {
                log.debug("Updating existing address for opportunity ID: {}", entity.getId());
                Address existingAddress = entity.getAddress();
                modelMapper.map(dto.getAddress(), existingAddress);
            } else {
                // Create a new address
                Address address = modelMapper.map(dto.getAddress(), Address.class);

                // Set default values if not provided
                if (address.getAddressType() == null) {
                    address.setAddressType(DEFAULT_ADDRESS_TYPE);
                }

                // Set up the bidirectional relationship properly
                entity.setAddress(address);

                // Manually manage the inverse relationship for validation
                if (address.getPartnershipOpportunities() == null) {
                    address.setPartnershipOpportunities(new ArrayList<>());
                }
                address.getPartnershipOpportunities().add(entity);
            }
        }
        // If no address provided in DTO but entity already has one, preserve it
        else if (entity.getAddress() != null && entity.getId() != null) {
            log.debug("Preserving existing address for opportunity ID: {}", entity.getId());
            // Address relationship is already established, no changes needed
        }
        // No address provided for a new entity
        else {
            log.warn("No address provided for partnership opportunity");
        }
    }

    /**
     * Updates an existing partnership opportunity from the given DTO.
     * <p>
     * Only allowed for admins or the owning company (if there are no active applications).
     *
     * @param id  the ID of the opportunity to update
     * @param dto the DTO containing updated data
     * @return the updated partnership opportunity
     * @throws AccessDeniedException if the user is not authorized to edit
     */
    @Override
    @Transactional(propagation = Propagation.REQUIRED)
    public PartnershipOpportunity update(Long id, PartnershipOpportunityDtoIn dto) {
        String userId = permissionUtils.getUserId();
        log.info("GDPR: Service=updateOpportunity, Operation=UPDATE_OPPORTUNITY, UserID={}, OpportunityID={}, Purpose=data_modification",
                userId, id);

        PartnershipOpportunity existingEntity = findById(id);
        if (permissionUtils.isAdmin() ||
                (permissionUtils.isCompany()
                        && permissionUtils.canEditOpportunity(existingEntity)
                        && !getSelf().hasActiveApplications(existingEntity))) {

            // Store reference to existing address and photos
            Address existingAddress = existingEntity.getAddress();
            List<PartnershipOpportunityPhoto> existingPhotos = new ArrayList<>(existingEntity.getPhotos());

            // Temporarily clear the address and photos to avoid mapping conflicts
            existingEntity.setAddress(null);
            existingEntity.getPhotos().clear();

            try {
                // Update entity properties manually instead of using ModelMapper for bulk mapping
                manuallyUpdateEntityFromDto(existingEntity, dto);

                // Handle address update separately
                if (dto.getAddressId() != null) {
                    log.debug("Updating address for partnership opportunity ID: {}", id);
                    processAddress(existingEntity, dto);
                } else if (dto.getAddress() != null) {
                    // Only update address if all required fields are provided
                    AddressDtoIn addressDto = dto.getAddress();
                    if (addressDto.getStreet() != null && !addressDto.getStreet().trim().isEmpty() &&
                            addressDto.getCity() != null && !addressDto.getCity().trim().isEmpty() &&
                            addressDto.getPostalCode() != null && !addressDto.getPostalCode().trim().isEmpty() &&
                            addressDto.getCountry() != null && !addressDto.getCountry().trim().isEmpty()) {
                        log.debug("Updating address for partnership opportunity ID: {}", id);
                        processAddress(existingEntity, dto);
                    } else if (existingAddress != null) {
                        // Incomplete address data - preserve existing address
                        log.debug("Preserving existing address due to incomplete data for opportunity ID: {}", id);
                        existingEntity.setAddress(existingAddress);
                        existingAddress.setPartnershipOpportunity(existingEntity);
                    }
                } else if (existingAddress != null) {
                    // If no address data provided in the DTO, preserve the existing address
                    log.debug("Preserving existing address for opportunity ID: {}", id);
                    existingEntity.setAddress(existingAddress);
                    existingAddress.setPartnershipOpportunity(existingEntity);
                }

                // Handle photos update
                if (dto.getPhotos() != null) {
                    log.debug("Updating photos for partnership opportunity ID: {}", id);
                    updatePhotosFromDto(existingEntity, dto.getPhotos(), existingPhotos);
                } else {
                    // Restore existing photos if no new photos provided
                    for (PartnershipOpportunityPhoto photo : existingPhotos) {
                        photo.setPartnershipOpportunity(existingEntity);
                    }
                    existingEntity.setPhotos(existingPhotos);
                }

                return repository.save(existingEntity);
            } catch (Exception e) {
                log.error("Error updating partnership opportunity: {}", e.getMessage(), e);
                // Restore entity state on error
                if (existingAddress != null) {
                    existingEntity.setAddress(existingAddress);
                }
                existingEntity.setPhotos(existingPhotos);
                throw e;
            }
        } else {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    userId,
                    "update",
                    "PartnershipOpportunity#" + id);
        }
    }

    /**
     * Manually updates the PartnershipOpportunity entity from the DTO without using ModelMapper.
     * This avoids the ModelMapper configuration issues with nested properties.
     *
     * @param entity The entity to update
     * @param dto    The DTO containing the updated values
     */
    private void manuallyUpdateEntityFromDto(PartnershipOpportunity entity, PartnershipOpportunityDtoIn dto) {
        // Update simple string fields
        if (dto.getName() != null) entity.setName(dto.getName());
        if (dto.getTitle() != null) entity.setTitle(dto.getTitle());
        if (dto.getDetails() != null) entity.setDetails(dto.getDetails());
        if (dto.getRequirements() != null) entity.setRequirements(dto.getRequirements());
        if (dto.getCompensationDescription() != null)
            entity.setCompensationDescription(dto.getCompensationDescription());

        // Update numeric fields
        entity.setCompensationAmountMin(dto.getCompensationAmountMin());
        entity.setCompensationAmountMax(dto.getCompensationAmountMax());
        entity.setFollowersMin(dto.getFollowersMin());
        entity.setFollowersMax(dto.getFollowersMax());

        // Update date fields
        if (dto.getStartDate() != null) entity.setStartDate(dto.getStartDate());
        if (dto.getEndDate() != null) entity.setEndDate(dto.getEndDate());

        // Update enum fields
        if (dto.getCompensationType() != null) entity.setCompensationType(dto.getCompensationType());

        // Update boolean fields
        entity.setActive(dto.isActive());

        // Handle relationships with repository lookups
        if (dto.getCity() != null) {
            // Get City repository from repositoryResolver and find by name
            try {
                CityRepository cityRepo = (CityRepository) repositoryResolver.getRepository(City.class);
                Optional<City> city = cityRepo.findByName(dto.getCity());
                if (city.isPresent()) {
                    entity.setCity(city.get());
                } else {
                    log.warn("City not found by name: {}", dto.getCity());
                }
            } catch (Exception e) {
                log.error("Error finding city by name: {}", dto.getCity(), e);
            }
        }

        if (dto.getCurrency() != null) {
            try {
                BaseRepository<Currency, Long> currencyRepo =
                        (BaseRepository<Currency, Long>) repositoryResolver.getRepository(Currency.class);

                Optional<Currency> currency = currencyRepo.findById(dto.getCurrency());
                if (currency.isPresent()) {
                    entity.setCurrency(currency.get());
                } else {
                    log.warn("Currency not found by ID: {}", dto.getCurrency());
                }
            } catch (Exception e) {
                log.error("Error finding currency by ID: {}", dto.getCurrency(), e);
            }
        }

        if (dto.getCompany() != null) {
            try {
                BaseRepository<User, Long> userRepo =
                        (BaseRepository<User, Long>) repositoryResolver.getRepository(User.class);

                Optional<User> user = userRepo.findById(dto.getCompany());
                if (user.isPresent()) {
                    entity.setCompany(user.get());
                } else {
                    log.warn("User not found by ID: {}", dto.getCompany());
                }
            } catch (Exception e) {
                log.error("Error finding user by ID: {}", dto.getCompany(), e);
            }
        }

        if (dto.getServiceType() != null) {
            try {
                BaseRepository<ServiceType, Long> serviceTypeRepo =
                        (BaseRepository<ServiceType, Long>) repositoryResolver.getRepository(ServiceType.class);

                Optional<ServiceType> serviceType = serviceTypeRepo.findById(dto.getServiceType());
                if (serviceType.isPresent()) {
                    entity.setServiceType(serviceType.get());
                } else {
                    log.warn("ServiceType not found by ID: {}", dto.getServiceType());
                }
            } catch (Exception e) {
                log.error("Error finding service type by ID: {}", dto.getServiceType(), e);
            }
        }

        // Handle collections
        if (dto.getPlatforms() != null && !dto.getPlatforms().isEmpty()) {
            Set<Platform> platforms = new HashSet<>();

            try {
                BaseRepository<Platform, Long> platformRepo =
                        (BaseRepository<Platform, Long>) repositoryResolver.getRepository(Platform.class);

                for (Long platformId : dto.getPlatforms()) {
                    Optional<Platform> platform = platformRepo.findById(platformId);
                    platform.ifPresent(platforms::add);
                }

                if (!platforms.isEmpty()) {
                    entity.setPlatforms(platforms);
                }
            } catch (Exception e) {
                log.error("Error finding platforms by IDs: {}", dto.getPlatforms(), e);
            }
        }

        if (dto.getContentTypes() != null && !dto.getContentTypes().isEmpty()) {
            Set<ContentType> contentTypes = new HashSet<>();

            try {
                BaseRepository<ContentType, Long> contentTypeRepo =
                        (BaseRepository<ContentType, Long>) repositoryResolver.getRepository(ContentType.class);

                for (Long contentTypeId : dto.getContentTypes()) {
                    Optional<ContentType> contentType = contentTypeRepo.findById(contentTypeId);
                    contentType.ifPresent(contentTypes::add);
                }

                if (!contentTypes.isEmpty()) {
                    entity.setContentTypes(contentTypes);
                }
            } catch (Exception e) {
                log.error("Error finding content types by IDs: {}", dto.getContentTypes(), e);
            }
        }
    }

    /**
     * Updates photos for a partnership opportunity based on the DTO input.
     * Preserves existing photos if they're in the DTO, updates changed ones, and adds new ones.
     *
     * @param entity         The partnership opportunity entity
     * @param photoDtos      The list of photo DTOs from the input
     * @param existingPhotos The existing photos in the entity
     */
    private void updatePhotosFromDto(PartnershipOpportunity entity,
                                     List<PartnershipOpportunityPhotoDtoIn> photoDtos,
                                     List<PartnershipOpportunityPhoto> existingPhotos) {
        // Create a map of existing photos by ID for efficient lookups
        Map<Long, PartnershipOpportunityPhoto> existingPhotosById = existingPhotos.stream()
                .filter(p -> p.getId() != null)
                .collect(Collectors.toMap(PartnershipOpportunityPhoto::getId, p -> p));

        // Track which photos are being kept
        Set<Long> processedPhotoIds = new HashSet<>();
        List<PartnershipOpportunityPhoto> updatedPhotos = new ArrayList<>();

        // Process each photo in the DTO
        for (PartnershipOpportunityPhotoDtoIn photoDto : photoDtos) {
            PartnershipOpportunityPhoto photo;

            // Check if this is an existing photo
            if (photoDto.getId() != null) {
                if (!existingPhotosById.containsKey(photoDto.getId())) {
                    // An id that isn't one of this campaign's photos is a
                    // client error — never silently create from it (the DTO
                    // carries no URL to create from anyway).
                    throw new ResourceNotFoundException("error.business.item_not_found", "Photo");
                }
                // Keep-by-id: the stored URL is BE-owned and never touched.
                photo = existingPhotosById.get(photoDto.getId());
                processedPhotoIds.add(photo.getId());
            } else {
                // New photo — must reference a tracked upload; the URL is
                // BE-derived (pentest 3.1 / 2026-06-13 hardening).
                photo = new PartnershipOpportunityPhoto();
                photo.setPartnershipOpportunity(entity);
                photo.setUrl(resolveNewPhotoUrl(photoDto.getUploadId()));
            }

            // Copy presentation fields explicitly (same idiom as
            // updatePhotoFromMap). Never ModelMapper here: with no explicit
            // type map, implicit STANDARD matching maps photoDto.id onto the
            // destination path photo.partnershipOpportunity.id (the source
            // class name PartnershipOpportunity*Photo*DtoIn supplies the
            // parent tokens), overwriting the attached parent's identifier —
            // Hibernate then fails the flush with "identifier of an instance
            // of PartnershipOpportunity was altered from <photoId> to <entityId>".
            photo.setOrderNumber(photoDto.getOrderNumber());
            photo.setIsCover(photoDto.getIsCover());

            updatedPhotos.add(photo);
        }

        // Set the updated photos on the entity
        entity.setPhotos(updatedPhotos);
    }

    /**
     * Checks if any application exists for the given opportunity and set of statuses.
     *
     * @param opportunityId the ID of the opportunity
     * @param statuses      the set of statuses to check against
     * @return true if matching applications exist, false otherwise
     */
    public boolean existsByOpportunityIdAndStatusIn(Long opportunityId, Set<OpportunityStatus> statuses) {
        return appliedOpportunityRepository.existsByPartnershipOpportunity_IdAndOpportunityStatusIn(opportunityId, statuses);
    }

    /**
     * Determines if a partnership opportunity has any active applications.
     *
     * @param po the partnership opportunity to check
     * @return true if active applications are present, false otherwise
     */
    public boolean hasActiveApplications(PartnershipOpportunity po) {

        return existsByOpportunityIdAndStatusIn(
                po.getId(),
                Set.of(OpportunityStatus.ACCEPTED_BY_COMPANY,
                        OpportunityStatus.CONTENT_APPROVED, OpportunityStatus.CONTENT_POSTED,
                        OpportunityStatus.CONTENT_REJECTED, OpportunityStatus.ACCEPTED_BY_INFLUENCER)
        );
    }

    /**
     * Applies partial updates to an existing partnership opportunity.
     * <p>
     * Fields like {@code photos} are handled specially. Restricted fields like {@code id}, {@code createdTime},
     * and {@code lastUpdateTime} are ignored.
     *
     * @param id      the ID of the opportunity to patch
     * @param updates a map of field names and new values
     * @return the updated partnership opportunity
     * @throws AccessDeniedException if the user is not allowed to patch the entity
     */
    @Override
    @Transactional(propagation = Propagation.REQUIRED)
    public PartnershipOpportunity patch(Long id, Map<String, Object> updates) {
        String userId = permissionUtils.getUserId();
        log.info("GDPR: Service=patchOpportunity, Operation=PARTIAL_UPDATE, UserID={}, OpportunityID={}, UpdatedFields={}, Purpose=partial_modification",
                userId, id, updates.keySet());

        PartnershipOpportunity existingEntity = findById(id);
        if (permissionUtils.isAdmin() ||
                (permissionUtils.isCompany()
                        && permissionUtils.canEditOpportunity(existingEntity)
                        && !getSelf().hasActiveApplications(existingEntity))) {

            // Process special fields first (address, photos) that need custom handling
            handleSpecialFieldUpdates(existingEntity, updates);

            // Process regular fields using reflection
            for (Map.Entry<String, Object> entry : updates.entrySet()) {
                String fieldName = entry.getKey();
                Object value = entry.getValue();

                // Skip fields that are handled specially
                if (fieldName.equals(ADDRESS) || fieldName.equals(ADDRESS_ID) ||
                        fieldName.equals(PHOTOS) || fieldName.equals("id") ||
                        fieldName.equals("createdTime") || fieldName.equals("lastUpdateTime")) {
                    continue;
                }

                try {
                    Field field = existingEntity.getClass().getDeclaredField(fieldName);
                    field.setAccessible(true);
                    Object castValue = convertValueToFieldType(field, value);
                    field.set(existingEntity, castValue);
                } catch (NoSuchFieldException e) {
                    log.warn("Field {} does not exist in class {}", fieldName, existingEntity.getClass().getSimpleName());
                    // Skip non-existent fields instead of throwing exception
                } catch (IllegalAccessException e) {
                    throw new InsufficientPermissionsException(
                            "error.auth.insufficient_permissions",
                            userId,
                            "patch." + fieldName,
                            "PartnershipOpportunity#" + id);
                }
            }

            return repository.save(getSelf().updateEntityUpdater(existingEntity));
        } else {
            throw new InsufficientPermissionsException(
                    "error.auth.insufficient_permissions",
                    userId,
                    "patch",
                    "PartnershipOpportunity#" + id);
        }
    }

    /**
     * Handles special fields that need custom processing logic during patch operations.
     *
     * @param entity  The entity to update
     * @param updates The map of updates to apply
     */
    private void handleSpecialFieldUpdates(PartnershipOpportunity entity, Map<String, Object> updates) {
        // Handle address updates
        if (updates.containsKey(ADDRESS_ID)) {
            Object value = updates.get(ADDRESS_ID);
            Long addressId = Long.valueOf(value.toString());
            Address address = addressRepository.findById(addressId)
                    .orElseThrow(() -> new ResourceNotFoundException("Address not found with ID: " + addressId));

            // Ensure address is not already associated with a user
            if (address.getUser() != null) {
                throw new IllegalArgumentException("Address is already associated with a user");
            }

            // Set the relationship in both directions
            address.setPartnershipOpportunity(entity);
            entity.setAddress(address);
        } else if (updates.containsKey(ADDRESS)) {
            Object addressValue = updates.get(ADDRESS);
            if (addressValue instanceof Map<?, ?> addressMap) {
                log.debug("Processing address from patch data: {}", addressMap);
                AddressDtoIn addressDto = convertToAddressDto(addressMap);

                // If the entity already has an address, update it
                if (entity.getAddress() != null) {
                    log.debug("Updating existing address for opportunity ID: {}", entity.getId());
                    Address existingAddress = entity.getAddress();
                    modelMapper.map(addressDto, existingAddress);
                } else {
                    // Create a new address
                    log.debug("Creating new address for opportunity ID: {}", entity.getId());
                    Address newAddress = modelMapper.map(addressDto, Address.class);
                    newAddress.setPartnershipOpportunity(entity);
                    entity.setAddress(newAddress);
                }
            } else {
                throw new IllegalArgumentException("Address data must be a map of properties, but was: " +
                        (addressValue != null ? addressValue.getClass().getName() : "null"));
            }
        }

        // Handle photos update
        if (updates.containsKey(PHOTOS)) {
            handlePhotoUpdates(entity, updates.get(PHOTOS));
        }
    }

    /**
     * Processes photo updates for a partnership opportunity.
     * This method handles adding, updating, and removing photos based on the provided data.
     *
     * @param entity      The partnership opportunity entity to update
     * @param photosValue The raw photo data from the update payload
     */
    private void handlePhotoUpdates(PartnershipOpportunity entity, Object photosValue) {
        log.debug("Processing photo updates for opportunity ID: {}", entity.getId());

        if (!(photosValue instanceof List<?> photosList)) {
            throw new IllegalArgumentException("Photos should be a list");
        }

        // Validate photo count limit
        if (photosList.size() > MAX_PHOTOS) {
            throw new ValidationTranslatableException("validation.photos.max", MAX_PHOTOS);
        }

        // Create a map of existing photos by ID for efficient lookups
        Map<Long, PartnershipOpportunityPhoto> existingPhotosById = entity.getPhotos().stream()
                .filter(p -> p.getId() != null)
                .collect(Collectors.toMap(PartnershipOpportunityPhoto::getId, p -> p));

        log.debug("Found {} existing photos", existingPhotosById.size());

        // Create a list to track which photos we're keeping
        Set<Long> processedPhotoIds = new HashSet<>();
        List<PartnershipOpportunityPhoto> updatedPhotos = new ArrayList<>();

        // Process each photo in the incoming list
        for (Object item : photosList) {
            if (!(item instanceof Map<?, ?>)) {
                throw new IllegalArgumentException("Each photo should be a map of properties");
            }

            // Convert to validated map with string keys
            Map<String, Object> validatedPhotoMap = new HashMap<>();
            for (Map.Entry<?, ?> entry : ((Map<?, ?>) item).entrySet()) {
                if (!(entry.getKey() instanceof String)) {
                    throw new IllegalArgumentException("Map keys must be strings");
                }
                validatedPhotoMap.put((String) entry.getKey(), entry.getValue());
            }

            PartnershipOpportunityPhoto photo;

            // Check if this is an existing photo (has ID and ID exists in our entity)
            if (validatedPhotoMap.containsKey("id") && validatedPhotoMap.get("id") != null) {
                Long photoId = Long.valueOf(validatedPhotoMap.get("id").toString());
                processedPhotoIds.add(photoId);

                // If it's an existing photo, update it
                if (existingPhotosById.containsKey(photoId)) {
                    log.debug("Updating existing photo with ID: {}", photoId);
                    photo = existingPhotosById.get(photoId);
                    updatePhotoFromMap(photo, validatedPhotoMap);
                } else {
                    // If ID is provided but doesn't exist in our database, throw error
                    throw new ResourceNotFoundException("Photo with ID " + photoId + " not found");
                }
            } else {
                // If no ID, it's a new photo — must reference a tracked
                // upload; the URL is BE-derived (pentest 3.1 / 2026-06-13).
                log.debug("Creating new photo");
                photo = new PartnershipOpportunityPhoto();
                Object uploadId = validatedPhotoMap.get("uploadId");
                photo.setUrl(resolveNewPhotoUrl(uploadId == null ? null : uploadId.toString()));
                updatePhotoFromMap(photo, validatedPhotoMap);
                photo.setPartnershipOpportunity(entity);
            }

            updatedPhotos.add(photo);
        }

        // Check if we need to remove any photos
        List<PartnershipOpportunityPhoto> photosToRemove = entity.getPhotos().stream()
                .filter(p -> p.getId() != null && !processedPhotoIds.contains(p.getId()))
                .toList();

        if (!photosToRemove.isEmpty()) {
            log.debug("Removing {} photos that were not in the update list", photosToRemove.size());
            // Actually remove the photos from the updated list
            updatedPhotos.removeAll(photosToRemove);
        }

        // Update the entity's photos list
        entity.setPhotos(updatedPhotos);
    }

    /**
     * Updates a photo entity with values from a map.
     *
     * @param photo    The photo entity to update
     * @param photoMap The map containing the new values
     */
    private void updatePhotoFromMap(PartnershipOpportunityPhoto photo, Map<String, Object> photoMap) {
        // NOTE: no "url" handling — the stored URL is BE-owned. New photos
        // get their URL from resolveNewPhotoUrl(uploadId); existing photos'
        // URLs are immutable through the PATCH surface (pentest 3.1).

        if (photoMap.containsKey(ORDER_NUMBER)) {
            photo.setOrderNumber((Integer) photoMap.get(ORDER_NUMBER));
        }

        if (photoMap.containsKey(IS_COVER)) {
            photo.setIsCover((Boolean) photoMap.get(IS_COVER));
        }
    }

    /**
     * Deactivates a Partnership Opportunity by its ID.
     * <p>
     * This method marks the opportunity as inactive instead of deleting it from the database.
     * It validates that the opportunity has no active applications and is currently active.
     *
     * @param id the ID of the Partnership Opportunity to deactivate
     * @throws BusinessRuleTranslatableException if the opportunity has active applications
     *                                           or is already inactive
     */
    @Override
    @Transactional(propagation = Propagation.REQUIRED)
    public void delete(Long id) {
        String userId = permissionUtils.getUserId();
        log.warn("GDPR: Service=deleteOpportunity, Operation=DEACTIVATE_OPPORTUNITY, UserID={}, OpportunityID={}, Purpose=data_deactivation",
                userId, id);

        PartnershipOpportunity partnershipOpportunity = findById(id);
        if (permissionUtils.isAdmin() ||
                (permissionUtils.isCompany()
                        && permissionUtils.canEditOpportunity(partnershipOpportunity)))
            deactivateOpportunity(partnershipOpportunity);
    }

    /**
     * Deactivates the given Partnership Opportunity.
     * <p>
     * Throws an exception if the opportunity is already inactive
     * or has active applications.
     *
     * @param po the Partnership Opportunity entity to deactivate
     * @throws BusinessRuleTranslatableException if the opportunity has active applications
     *                                           or is already inactive
     */
    private void deactivateOpportunity(PartnershipOpportunity po) {

        if (getSelf().hasActiveApplications(po))
            throw new BusinessRuleTranslatableException("error.business.rule_violation");
        if (!po.isActive()) {
            throw new BusinessRuleTranslatableException("error.business.invalid_state");
        }
        po.setActive(false);
        save(po);
    }

    /**
     * Converts a DTO object into a map of its fields and values.
     * <p>
     * Handles nested lists and custom types like {@code PartnershipOpportunityPhoto}.
     *
     * @param dto the DTO object to convert
     * @return a map representing the DTO's fields
     */
    private Map<String, Object> convertDtoToMap(Object dto) {
        Map<String, Object> map = new HashMap<>();
        try {
            Field[] fields = dto.getClass().getDeclaredFields();
            for (Field field : fields) {
                field.setAccessible(true);
                String fieldName = field.getName();
                Object fieldValue = field.get(dto);

                if (fieldValue instanceof List<?> listValue) {
                    if (!listValue.isEmpty() && listValue.getFirst() instanceof PartnershipOpportunityPhoto) {
                        List<Map<String, Object>> photosMap = new ArrayList<>();
                        for (Object item : listValue) {
                            PartnershipOpportunityPhoto photo = (PartnershipOpportunityPhoto) item;
                            Map<String, Object> photoMap = new HashMap<>();
                            photoMap.put("id", photo.getId());
                            photoMap.put("url", photo.getUrl());
                            photoMap.put(ORDER_NUMBER, photo.getOrderNumber());
                            photoMap.put(IS_COVER, photo.getIsCover());
                            photosMap.add(photoMap);
                        }
                        map.put(fieldName, photosMap);
                    } else {
                        List<Map<String, Object>> listMap = new ArrayList<>();
                        for (Object item : listValue) {
                            listMap.add(convertDtoToMap(item));
                        }
                        map.put(fieldName, listMap);
                    }
                } else if (fieldValue != null) {
                    map.put(fieldName, fieldValue);
                }
            }
        } catch (IllegalAccessException e) {
            throw new BusinessRuleTranslatableException("error.business.invalid_state");
        }
        return map;
    }

    /**
     * Converts a map of address properties to an AddressDtoIn object.
     *
     * @param addressMap The map containing address properties
     * @return An AddressDtoIn populated with the map values
     */
    private AddressDtoIn convertToAddressDto(Map<?, ?> addressMap) {
        AddressDtoIn addressDto = new AddressDtoIn();

        // Convert map to a validated map with string keys
        Map<String, Object> validatedMap = new HashMap<>();
        for (Map.Entry<?, ?> entry : addressMap.entrySet()) {
            if (!(entry.getKey() instanceof String)) {
                throw new ValidationTranslatableException("error.validation.invalid_structure");
            }
            validatedMap.put((String) entry.getKey(), entry.getValue());
        }

        // Set ID if present
        if (validatedMap.containsKey("id") && validatedMap.get("id") != null) {
            addressDto.setId(Long.valueOf(validatedMap.get("id").toString()));
        }

        // Set address properties
        if (validatedMap.containsKey("street")) {
            addressDto.setStreet((String) validatedMap.get("street"));
        }

        if (validatedMap.containsKey("city")) {
            addressDto.setCity((String) validatedMap.get("city"));
        }

        if (validatedMap.containsKey("postalCode")) {
            addressDto.setPostalCode((String) validatedMap.get("postalCode"));
        }

        if (validatedMap.containsKey("country")) {
            addressDto.setCountry((String) validatedMap.get("country"));
        }

        if (validatedMap.containsKey("state")) {
            addressDto.setState((String) validatedMap.get("state"));
        }

        if (validatedMap.containsKey("additionalInfo")) {
            addressDto.setAdditionalInfo((String) validatedMap.get("additionalInfo"));
        }

        if (validatedMap.containsKey(ADDRESS_TYPE) && validatedMap.get(ADDRESS_TYPE) != null) {
            addressDto.setAddressType(validatedMap.get(ADDRESS_TYPE).toString());
        } else {
            addressDto.setAddressType(DEFAULT_ADDRESS_TYPE);
        }

        if (validatedMap.containsKey(IS_PRIMARY) && validatedMap.get(IS_PRIMARY) != null) {
            addressDto.setPrimary((Boolean) validatedMap.get(IS_PRIMARY));
        }

        return addressDto;
    }
}