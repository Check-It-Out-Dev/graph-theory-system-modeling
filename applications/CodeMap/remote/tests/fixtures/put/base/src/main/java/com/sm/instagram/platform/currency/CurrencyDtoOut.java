package com.sm.instagram.platform.currency;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Output DTO for Currency with translation support
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CurrencyDtoOut {
    private Long id;
    private String name;         // Translated name
    private String originalName; // Original name from DB
    private String isoCode;
    private String sign;
    private String countryCode;
}
