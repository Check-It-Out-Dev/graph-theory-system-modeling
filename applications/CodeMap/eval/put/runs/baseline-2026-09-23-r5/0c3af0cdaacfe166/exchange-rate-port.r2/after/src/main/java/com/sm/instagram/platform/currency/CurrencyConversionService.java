package com.sm.instagram.platform.currency;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Converts amounts in other currencies to PLN using whatever {@link ExchangeRatePort} is wired in,
 * so the rate source can be swapped without touching the conversion itself.
 */
@Service
public class CurrencyConversionService {

    private final ExchangeRatePort exchangeRatePort;

    public CurrencyConversionService(ExchangeRatePort exchangeRatePort) {
        this.exchangeRatePort = exchangeRatePort;
    }

    public BigDecimal toPln(BigDecimal amount, String isoCode) {
        BigDecimal rate = exchangeRatePort.rateToPln(isoCode)
                .orElseThrow(() -> new BusinessRuleTranslatableException("error.business.exchange_rate_unavailable", isoCode));

        return amount.multiply(rate).setScale(2, RoundingMode.HALF_UP);
    }
}
