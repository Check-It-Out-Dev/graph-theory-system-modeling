package com.sm.instagram.platform.currency;

import com.sm.instagram.platform.common.base.BaseService;
import com.sm.instagram.platform.common.util.RepositoryResolver;
import com.sm.instagram.platform.common.util.filtering.SpecificationBuilder;
import lombok.extern.slf4j.Slf4j;
import org.modelmapper.ModelMapper;
import org.springframework.context.ApplicationContext;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.Map;
import java.util.Optional;

@Slf4j
@Service
public class CurrencyService extends BaseService<Currency, Long, CurrencyDto> {

    private final CurrencyRepository currencyRepository;

    protected CurrencyService(SpecificationBuilder<Currency> specificationBuilder,
                              CurrencyRepository repository,
                              ModelMapper modelMapper,
                              RepositoryResolver repositoryResolver,
                              ApplicationContext applicationContext) {
        super(applicationContext, specificationBuilder, repository, modelMapper, repositoryResolver);
        this.currencyRepository = repository;
    }

    /**
     * Find currency by ISO code
     *
     * @param isoCode The ISO code to search for
     * @return Optional containing the currency if found
     */
    public Optional<Currency> findByIsoCode(String isoCode) {
        String firebaseUid = "SYSTEM";
        try {
            firebaseUid = SecurityContextHolder.getContext().getAuthentication().getPrincipal().toString();
        } catch (Exception e) {
            // Use SYSTEM if no auth context
        }

        log.info("GDPR: Operation=findCurrencyByIsoCode, FirebaseUID={}, IsoCode={}, Purpose=reference_data_lookup",
                firebaseUid, isoCode);

        Optional<Currency> currency = currencyRepository.findByIsoCode(isoCode);

        if (currency.isPresent()) {
            log.info("GDPR: DataAccessed=currency, FirebaseUID={}, CurrencyID={}, Purpose=iso_code_retrieval",
                    firebaseUid, currency.get().getId());
        }

        return currency;
    }

    /**
     * Converts a Currency entity to CurrencyDtoOut.
     * Must be called within a transaction to avoid LazyInitializationException.
     *
     * @param entity The Currency entity to convert
     * @return CurrencyDtoOut or null if entity is null
     */
    @Override
    @Transactional(readOnly = true)
    public <DTOOUT> DTOOUT toDto(Currency entity) {
        if (entity == null) return null;

        @SuppressWarnings("unchecked")
        DTOOUT result = (DTOOUT) modelMapper.map(entity, CurrencyDtoOut.class);
        return result;
    }

    /**
     * Creates a new Currency from DTO and returns as DTO.
     * All conversions happen within transaction boundary.
     *
     * @param dto The input DTO
     * @return The created entity as DTO
     */
    @Override
    @Transactional
    public <DTOOUT> DTOOUT createFromDtoAsDto(CurrencyDto dto) {
        String firebaseUid = "SYSTEM";
        try {
            firebaseUid = SecurityContextHolder.getContext().getAuthentication().getPrincipal().toString();
        } catch (Exception e) {
            // Use SYSTEM if no auth context
        }

        log.info("GDPR: Operation=createCurrency, FirebaseUID={}, CurrencyName={}, IsoCode={}, Purpose=reference_data_creation",
                firebaseUid, dto.getName(), dto.getIsoCode());

        Currency entity = modelMapper.map(dto, Currency.class);
        Currency saved = save(entity);

        log.info("GDPR: DataCreated=currency, FirebaseUID={}, CurrencyID={}, Purpose=reference_data",
                firebaseUid, saved.getId());

        return toDto(saved);
    }

    /**
     * Retrieves a paginated list of entities as DTOs with all conversions done within transaction.
     * This prevents LazyInitializationException by ensuring all DTO mappings happen inside @Transactional.
     *
     * @param pageable Pagination parameters
     * @param filters  Filter parameters
     * @param <DTOOUT> The output DTO type
     * @return Page of DTOs with all lazy relationships properly loaded
     */
    @Override
    public <DTOOUT> Page<DTOOUT> getDataPagedAndFilteredAsDtos(Pageable pageable, Map<String, String> filters) {
        String firebaseUid = "SYSTEM";
        try {
            firebaseUid = SecurityContextHolder.getContext().getAuthentication().getPrincipal().toString();
        } catch (Exception e) {
            // Use SYSTEM if no auth context
        }

        log.info("GDPR: Operation=getCurrenciesPaged, FirebaseUID={}, Purpose=reference_data_retrieval", firebaseUid);

        // Get entities with proper pagination and filtering
        Page<Currency> page = getDataPagedAndFiltered(pageable, filters);

        // Convert to DTOs within transaction boundary
        Page<CurrencyDtoOut> dtoPage = page.map(entity -> {
            CurrencyDtoOut dto = new CurrencyDtoOut();

            // Map basic fields
            dto.setId(entity.getId());
            dto.setName(entity.getName());
            dto.setOriginalName(entity.getName()); // Store original name for translation reference
            dto.setIsoCode(entity.getIsoCode());
            dto.setSign(entity.getSign());
            dto.setCountryCode(entity.getCountryCode());

            return dto;
        });

        // Cast to generic type
        @SuppressWarnings("unchecked")
        Page<DTOOUT> result = (Page<DTOOUT>) dtoPage;

        log.info("GDPR: DataAccessed=currencies, FirebaseUID={}, RecordCount={}, Purpose=reference_data_listing",
                firebaseUid, result.getTotalElements());

        return result;
    }
}
