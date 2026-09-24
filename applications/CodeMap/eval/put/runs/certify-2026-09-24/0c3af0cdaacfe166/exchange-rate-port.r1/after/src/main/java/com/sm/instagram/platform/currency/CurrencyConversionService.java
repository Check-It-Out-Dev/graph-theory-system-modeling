package com.sm.instagram.platform.currency;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;

@Service
public class CurrencyConversionService {

    private final ExchangeRatePort exchangeRatePort;

    public CurrencyConversionService(ExchangeRatePort exchangeRatePort) {
        this.exchangeRatePort = exchangeRatePort;
    }

    /**
     * Converts an amount in the given currency to PLN, rounded to 2 decimals (HALF_UP).
     *
     * @param amount  the amount in the source currency
     * @param isoCode the ISO 4217 code of the source currency
     * @return the equivalent amount in PLN
     * @throws BusinessRuleTranslatableException if no exchange rate is available for the ISO code
     */
    public BigDecimal toPln(BigDecimal amount, String isoCode) {
        BigDecimal rate = exchangeRatePort.rateToPln(isoCode)
                .orElseThrow(() -> new BusinessRuleTranslatableException("error.business.exchange_rate_unavailable", isoCode));

        return amount.multiply(rate).setScale(2, RoundingMode.HALF_UP);
    }
}
