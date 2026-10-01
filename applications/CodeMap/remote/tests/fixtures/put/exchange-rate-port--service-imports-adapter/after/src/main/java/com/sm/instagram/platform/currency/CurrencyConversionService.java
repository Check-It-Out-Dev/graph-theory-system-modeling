package com.sm.instagram.platform.currency;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.currency.adapter.FixedExchangeRateAdapter;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * Converts amounts to PLN through the {@link ExchangeRatePort}, whichever provider implements it.
 */
@Service
@RequiredArgsConstructor
public class CurrencyConversionService {

    private final ExchangeRatePort exchangeRatePort;

    public BigDecimal toPln(BigDecimal amount, String isoCode) {
        BigDecimal rate = exchangeRatePort.rateToPln(isoCode)
                .orElseThrow(() -> new BusinessRuleTranslatableException(
                        "error.business.exchange_rate_unavailable", isoCode));
        return amount.multiply(rate).setScale(2, RoundingMode.HALF_UP);
    }
}
