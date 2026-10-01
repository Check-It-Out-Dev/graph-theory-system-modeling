package com.sm.instagram.platform.registry.port;

/**
 * Port interface for company data lookup from a business registry.
 * Primary implementation: {@link com.sm.instagram.platform.registry.adapter.gus.GusBir1RegistryAdapter} (SOAP).
 * Swappable by implementing this interface in a new adapter and using {@code @Primary} or {@code @Profile}.
 */
public interface CompanyRegistryPort {

    /**
     * Looks up a company by its NIP (tax identification number).
     *
     * @param nip 10-digit Polish NIP
     * @return company data from the registry
     * @throws com.sm.instagram.platform.common.exceptions.ExternalServiceException if the registry is unavailable
     */
    CompanyRegistryData lookupByNip(String nip);
}
