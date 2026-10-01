package com.sm.instagram.platform.currency;

import com.sm.instagram.platform.common.exceptions.BusinessRuleTranslatableException;
import com.sm.instagram.platform.currency.port.ExchangeRatePort;
import org.springframework.stereotype.Service;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Optional;

@Service
public class CurrencyConversionService {

    private final ExchangeRatePort exchangeRatePort;

    public CurrencyConversionService(ExchangeRatePort exchangeRatePort) {
        this.exchangeRatePort = exchangeRatePort;
    }

    public BigDecimal toPln(BigDecimal amount, String isoCode) {
        Optional<BigDecimal> rate = exchangeRatePort.rateToPln(isoCode);

        if (rate.isEmpty()) {
            throw new BusinessRuleTranslatableException("error.business.exchange_rate_unavailable", isoCode);
        }

        return amount.multiply(rate.get()).setScale(2, RoundingMode.HALF_UP);
    }
}
