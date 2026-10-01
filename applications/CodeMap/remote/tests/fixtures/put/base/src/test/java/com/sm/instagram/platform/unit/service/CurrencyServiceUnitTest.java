package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.currency.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

import jakarta.validation.Validation;
import jakarta.validation.Validator;
import jakarta.validation.ValidatorFactory;
import jakarta.validation.ConstraintViolation;

import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;

/**
 * Unit tests for Currency entity and DTOs.
 * Tests entity validation and basic DTO mapping logic.
 */
@DisplayName("Currency Unit Tests")
class CurrencyServiceUnitTest {

    private Validator validator;

    @BeforeEach
    void setUp() {
        ValidatorFactory factory = Validation.buildDefaultValidatorFactory();
        validator = factory.getValidator();
    }

    // ==================== Currency Entity Tests ====================

    @Nested
    @DisplayName("Currency Entity")
    class CurrencyEntityTests {

        @Test
        @DisplayName("should create valid currency with all fields")
        void shouldCreateValidCurrency() {
            // Given
            Currency currency = new Currency();
            currency.setId(1L);
            currency.setName("US Dollar");
            currency.setIsoCode("USD");
            currency.setSign("$");
            currency.setCountryCode("US");

            // Then
            assertThat(currency.getId()).isEqualTo(1L);
            assertThat(currency.getName()).isEqualTo("US Dollar");
            assertThat(currency.getIsoCode()).isEqualTo("USD");
            assertThat(currency.getSign()).isEqualTo("$");
            assertThat(currency.getCountryCode()).isEqualTo("US");
        }

        @Test
        @DisplayName("should create currency using all-args constructor")
        void shouldCreateCurrencyUsingAllArgsConstructor() {
            // Given
            Currency currency = new Currency(1L, "Euro", "EUR", "€", "EU");

            // Then
            assertThat(currency.getId()).isEqualTo(1L);
            assertThat(currency.getName()).isEqualTo("Euro");
            assertThat(currency.getIsoCode()).isEqualTo("EUR");
            assertThat(currency.getSign()).isEqualTo("€");
            assertThat(currency.getCountryCode()).isEqualTo("EU");
        }

        @Test
        @DisplayName("should create currency using no-args constructor")
        void shouldCreateCurrencyUsingNoArgsConstructor() {
            // Given
            Currency currency = new Currency();

            // Then
            assertThat(currency.getId()).isNull();
            assertThat(currency.getName()).isNull();
        }

        @ParameterizedTest
        @CsvSource({
                "USD, US Dollar, $, US",
                "EUR, Euro, €, EU",
                "PLN, Polish Złoty, zł, PL",
                "GBP, British Pound, £, GB",
                "JPY, Japanese Yen, ¥, JP"
        })
        @DisplayName("should store various currency data")
        void shouldStoreVariousCurrencyData(String isoCode, String name, String sign, String countryCode) {
            // Given
            Currency currency = new Currency();
            currency.setName(name);
            currency.setIsoCode(isoCode);
            currency.setSign(sign);
            currency.setCountryCode(countryCode);

            // Then
            assertThat(currency.getName()).isEqualTo(name);
            assertThat(currency.getIsoCode()).isEqualTo(isoCode);
            assertThat(currency.getSign()).isEqualTo(sign);
            assertThat(currency.getCountryCode()).isEqualTo(countryCode);
        }
    }

    // ==================== Currency Entity Validation Tests ====================

    @Nested
    @DisplayName("Currency Entity Validation")
    class CurrencyValidationTests {

        @Test
        @DisplayName("should pass validation for valid currency")
        void shouldPassValidationForValidCurrency() {
            // Given
            Currency currency = new Currency(null, "US Dollar", "USD", "$", "US");

            // When
            Set<ConstraintViolation<Currency>> violations = validator.validate(currency);

            // Then
            assertThat(violations).isEmpty();
        }

        @ParameterizedTest
        @NullAndEmptySource
        @ValueSource(strings = {" ", "  ", "\t"})
        @DisplayName("should fail validation for blank name")
        void shouldFailValidationForBlankName(String name) {
            // Given
            Currency currency = new Currency(null, name, "USD", "$", "US");

            // When
            Set<ConstraintViolation<Currency>> violations = validator.validate(currency);

            // Then
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("name"))).isTrue();
        }

        @ParameterizedTest
        @NullAndEmptySource
        @ValueSource(strings = {" ", "  ", "\t"})
        @DisplayName("should fail validation for blank ISO code")
        void shouldFailValidationForBlankIsoCode(String isoCode) {
            // Given
            Currency currency = new Currency(null, "US Dollar", isoCode, "$", "US");

            // When
            Set<ConstraintViolation<Currency>> violations = validator.validate(currency);

            // Then
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("isoCode"))).isTrue();
        }

        @ParameterizedTest
        @NullAndEmptySource
        @ValueSource(strings = {" ", "  ", "\t"})
        @DisplayName("should fail validation for blank sign")
        void shouldFailValidationForBlankSign(String sign) {
            // Given
            Currency currency = new Currency(null, "US Dollar", "USD", sign, "US");

            // When
            Set<ConstraintViolation<Currency>> violations = validator.validate(currency);

            // Then
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getPropertyPath().toString().equals("sign"))).isTrue();
        }

        @Test
        @DisplayName("should fail validation for ISO code longer than 3 characters")
        void shouldFailValidationForLongIsoCode() {
            // Given
            Currency currency = new Currency(null, "Dollar", "USDX", "$", "US");

            // When
            Set<ConstraintViolation<Currency>> violations = validator.validate(currency);

            // Then
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getMessage().contains("ISO code cannot exceed 3 characters"))).isTrue();
        }

        @Test
        @DisplayName("should fail validation for sign longer than 3 characters")
        void shouldFailValidationForLongSign() {
            // Given
            Currency currency = new Currency(null, "Dollar", "USD", "$$$$", "US");

            // When
            Set<ConstraintViolation<Currency>> violations = validator.validate(currency);

            // Then
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getMessage().contains("Sign cannot exceed 3 characters"))).isTrue();
        }

        @Test
        @DisplayName("should fail validation for country code longer than 3 characters")
        void shouldFailValidationForLongCountryCode() {
            // Given
            Currency currency = new Currency(null, "Dollar", "USD", "$", "USAX");

            // When
            Set<ConstraintViolation<Currency>> violations = validator.validate(currency);

            // Then
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getMessage().contains("country code cannot exceed 3 characters"))).isTrue();
        }

        @Test
        @DisplayName("should fail validation for name longer than 255 characters")
        void shouldFailValidationForLongName() {
            // Given
            String longName = "A".repeat(256);
            Currency currency = new Currency(null, longName, "USD", "$", "US");

            // When
            Set<ConstraintViolation<Currency>> violations = validator.validate(currency);

            // Then
            assertThat(violations).isNotEmpty();
            assertThat(violations.stream()
                    .anyMatch(v -> v.getMessage().contains("Name cannot exceed 255 characters"))).isTrue();
        }
    }

    // ==================== CurrencyDto Tests ====================

    @Nested
    @DisplayName("CurrencyDto")
    class CurrencyDtoTests {

        @Test
        @DisplayName("should create DTO with all fields")
        void shouldCreateDtoWithAllFields() {
            // Given
            CurrencyDto dto = new CurrencyDto();
            dto.setName("Polish Złoty");
            dto.setIsoCode("PLN");
            dto.setSign("zł");
            dto.setCountryCode("PL");

            // Then
            assertThat(dto.getName()).isEqualTo("Polish Złoty");
            assertThat(dto.getIsoCode()).isEqualTo("PLN");
            assertThat(dto.getSign()).isEqualTo("zł");
            assertThat(dto.getCountryCode()).isEqualTo("PL");
        }
    }

    // ==================== CurrencyDtoOut Tests ====================

    @Nested
    @DisplayName("CurrencyDtoOut")
    class CurrencyDtoOutTests {

        @Test
        @DisplayName("should create output DTO with all fields")
        void shouldCreateOutputDtoWithAllFields() {
            // Given
            CurrencyDtoOut dto = new CurrencyDtoOut();
            dto.setId(1L);
            dto.setName("Euro");
            dto.setOriginalName("Euro");
            dto.setIsoCode("EUR");
            dto.setSign("€");
            dto.setCountryCode("EU");

            // Then
            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getName()).isEqualTo("Euro");
            assertThat(dto.getOriginalName()).isEqualTo("Euro");
            assertThat(dto.getIsoCode()).isEqualTo("EUR");
            assertThat(dto.getSign()).isEqualTo("€");
            assertThat(dto.getCountryCode()).isEqualTo("EU");
        }

        @Test
        @DisplayName("should support translation with different name and original name")
        void shouldSupportTranslationWithDifferentNames() {
            // Given - Simulating translated name
            CurrencyDtoOut dto = new CurrencyDtoOut();
            dto.setName("Złoty polski");  // Translated name
            dto.setOriginalName("Polish Zloty");  // Original name
            dto.setIsoCode("PLN");

            // Then
            assertThat(dto.getName()).isEqualTo("Złoty polski");
            assertThat(dto.getOriginalName()).isEqualTo("Polish Zloty");
            assertThat(dto.getName()).isNotEqualTo(dto.getOriginalName());
        }
    }

    // ==================== Common Currency Values Tests ====================

    @Nested
    @DisplayName("Common Currency Values")
    class CommonCurrencyValuesTests {

        @ParameterizedTest
        @CsvSource({
                "USD, 840",
                "EUR, 978",
                "PLN, 985",
                "GBP, 826",
                "CHF, 756"
        })
        @DisplayName("should handle ISO 4217 standard codes")
        void shouldHandleIso4217StandardCodes(String isoCode, String numericCode) {
            // ISO codes should be valid
            assertThat(isoCode).hasSize(3);
            assertThat(isoCode).matches("[A-Z]{3}");
        }

        @Test
        @DisplayName("should handle special currency signs")
        void shouldHandleSpecialCurrencySigns() {
            // Given
            String[] signs = {"$", "€", "£", "¥", "zł", "₽", "₴", "₿"};

            // Then - All signs should be short
            for (String sign : signs) {
                assertThat(sign.length()).isLessThanOrEqualTo(3);
            }
        }
    }

    // ==================== Edge Cases ====================

    @Nested
    @DisplayName("Edge Cases")
    class EdgeCaseTests {

        @Test
        @DisplayName("should handle currency with maximum length name")
        void shouldHandleCurrencyWithMaxLengthName() {
            // Given
            String maxName = "A".repeat(255);
            Currency currency = new Currency(null, maxName, "USD", "$", "US");

            // When
            Set<ConstraintViolation<Currency>> violations = validator.validate(currency);

            // Then - should pass with exactly 255 characters
            assertThat(violations).isEmpty();
            assertThat(currency.getName()).hasSize(255);
        }

        @Test
        @DisplayName("should handle single character ISO code")
        void shouldHandleSingleCharIsoCode() {
            // Given - Edge case: very short ISO code
            Currency currency = new Currency(null, "Test", "X", "$", "X");

            // When
            Set<ConstraintViolation<Currency>> violations = validator.validate(currency);

            // Then - should pass (no minimum length constraint)
            assertThat(violations).isEmpty();
        }

        @Test
        @DisplayName("should handle single character sign")
        void shouldHandleSingleCharSign() {
            // Given
            Currency currency = new Currency(null, "Dollar", "USD", "$", "US");

            // When
            Set<ConstraintViolation<Currency>> violations = validator.validate(currency);

            // Then
            assertThat(violations).isEmpty();
            assertThat(currency.getSign()).hasSize(1);
        }

        @Test
        @DisplayName("should handle exactly 3 character values")
        void shouldHandleExactlyThreeCharacterValues() {
            // Given - All max length values
            Currency currency = new Currency(null, "Abc", "ABC", "ABC", "ABC");

            // When
            Set<ConstraintViolation<Currency>> violations = validator.validate(currency);

            // Then
            assertThat(violations).isEmpty();
            assertThat(currency.getIsoCode()).hasSize(3);
            assertThat(currency.getSign()).hasSize(3);
            assertThat(currency.getCountryCode()).hasSize(3);
        }
    }
}
