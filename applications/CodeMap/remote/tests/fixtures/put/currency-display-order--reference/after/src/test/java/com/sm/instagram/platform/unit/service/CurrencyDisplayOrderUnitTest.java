package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.currency.Currency;
import com.sm.instagram.platform.currency.CurrencyDtoOut;
import com.sm.instagram.platform.currency.CurrencyRepository;
import com.sm.instagram.platform.currency.CurrencyService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.Spy;
import org.mockito.junit.jupiter.MockitoExtension;
import org.modelmapper.ModelMapper;

import static org.assertj.core.api.Assertions.assertThat;

@ExtendWith(MockitoExtension.class)
class CurrencyDisplayOrderUnitTest {

    @Mock
    private CurrencyRepository currencyRepository;

    @Spy
    private ModelMapper modelMapper = new ModelMapper();

    @InjectMocks
    private CurrencyService currencyService;

    @Test
    void toDtoMapsTheDisplayOrder() {
        Currency pln = new Currency();
        pln.setIsoCode("PLN");
        pln.setDisplayOrder(1);

        CurrencyDtoOut dto = currencyService.toDto(pln);

        assertThat(dto.getDisplayOrder()).isEqualTo(1);
    }

    @Test
    void newCurrencyHasNoPlaceYet() {
        assertThat(new Currency().getDisplayOrder()).isZero();
    }
}
