package com.sm.instagram.platform.unit.service;

import com.sm.instagram.platform.activecooperations.CoopDto;
import com.sm.instagram.platform.activecooperations.CoopFilter;
import com.sm.instagram.platform.activecooperations.Views;
import com.sm.instagram.platform.address.AddressCityOnlyDto;
import com.sm.instagram.platform.address.AddressDtoOut;
import com.sm.instagram.platform.address.AddressSourceType;
import com.sm.instagram.platform.appliedopportunities.OpportunityStatus;
import com.sm.instagram.platform.appliedopportunities.RateStatus;
import com.sm.instagram.platform.currency.CurrencyDto;
import com.sm.instagram.platform.currency.CurrencyDtoOut;
import com.sm.instagram.platform.user.AccountStatusDtoOut;
import com.sm.instagram.platform.user.CompanyPublicProfileDto;
import com.sm.instagram.platform.user.InfluencerForCompanyProfileDto;
import com.sm.instagram.platform.user.InfluencerPublicProfileDto;
import com.sm.instagram.platform.user.PublicProfileDto;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.time.LocalDateTime;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

@DisplayName("Address, Currency & Profile DTO Unit Tests")
class AddressAndCurrencyUnitTest {

    @Nested
    @DisplayName("AddressSourceType Enum Tests")
    class AddressSourceTypeTests {

        @Test
        @DisplayName("should have 3 values")
        void shouldHave3Values() {
            assertThat(AddressSourceType.values()).hasSize(3);
        }

        @ParameterizedTest
        @EnumSource(AddressSourceType.class)
        @DisplayName("all values should have name")
        void allValuesShouldHaveName(AddressSourceType type) {
            assertThat(type.name()).isNotBlank();
        }

        @Test
        @DisplayName("COPIED_FROM_USER should exist")
        void copiedFromUserShouldExist() {
            AddressSourceType type = AddressSourceType.valueOf("COPIED_FROM_USER");
            assertThat(type).isEqualTo(AddressSourceType.COPIED_FROM_USER);
        }

        @Test
        @DisplayName("BUSINESS_LOCATION should exist")
        void businessLocationShouldExist() {
            AddressSourceType type = AddressSourceType.valueOf("BUSINESS_LOCATION");
            assertThat(type).isEqualTo(AddressSourceType.BUSINESS_LOCATION);
        }

        @Test
        @DisplayName("CUSTOM should exist")
        void customShouldExist() {
            AddressSourceType type = AddressSourceType.valueOf("CUSTOM");
            assertThat(type).isEqualTo(AddressSourceType.CUSTOM);
        }

        @Test
        @DisplayName("ordinal values should be correct")
        void ordinalValuesShouldBeCorrect() {
            assertThat(AddressSourceType.COPIED_FROM_USER.ordinal()).isZero();
            assertThat(AddressSourceType.BUSINESS_LOCATION.ordinal()).isEqualTo(1);
            assertThat(AddressSourceType.CUSTOM.ordinal()).isEqualTo(2);
        }
    }

    @Nested
    @DisplayName("AddressCityOnlyDto Tests")
    class AddressCityOnlyDtoTests {

        @Test
        @DisplayName("getter and setter should work")
        void getterAndSetterShouldWork() {
            AddressCityOnlyDto dto = new AddressCityOnlyDto();
            dto.setCity("Warsaw");

            assertThat(dto.getCity()).isEqualTo("Warsaw");
        }

        @Test
        @DisplayName("should accept null city")
        void shouldAcceptNullCity() {
            AddressCityOnlyDto dto = new AddressCityOnlyDto();
            dto.setCity(null);

            assertThat(dto.getCity()).isNull();
        }
    }

    @Nested
    @DisplayName("AddressDtoOut Tests")
    class AddressDtoOutTests {

        @Test
        @DisplayName("no-args constructor should initialize defaults")
        void noArgsConstructorShouldInitializeDefaults() {
            AddressDtoOut dto = new AddressDtoOut();

            assertThat(dto.getId()).isNull();
            assertThat(dto.isPrimary()).isFalse();
            assertThat(dto.isShared()).isFalse();
            assertThat(dto.isCopied()).isFalse();
        }

        @Test
        @DisplayName("all-args constructor should set all fields")
        void allArgsConstructorShouldSetAllFields() {
            LocalDateTime now = LocalDateTime.now();

            AddressDtoOut dto = new AddressDtoOut(
                    1L, 10L, "Main Street 1", "Warsaw", "00-001", "Poland", "Mazowieckie",
                    "Apt 5", "HOME", true, now, now, AddressSourceType.BUSINESS_LOCATION,
                    true, "admin", true, 5L, 100L
            );

            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getUserId()).isEqualTo(10L);
            assertThat(dto.getStreet()).isEqualTo("Main Street 1");
            assertThat(dto.getCity()).isEqualTo("Warsaw");
            assertThat(dto.getPostalCode()).isEqualTo("00-001");
            assertThat(dto.getCountry()).isEqualTo("Poland");
            assertThat(dto.getState()).isEqualTo("Mazowieckie");
            assertThat(dto.getAdditionalInfo()).isEqualTo("Apt 5");
            assertThat(dto.getAddressType()).isEqualTo("HOME");
            assertThat(dto.isPrimary()).isTrue();
            assertThat(dto.getCreatedTime()).isEqualTo(now);
            assertThat(dto.getLastUpdateTime()).isEqualTo(now);
            assertThat(dto.getSourceType()).isEqualTo(AddressSourceType.BUSINESS_LOCATION);
            assertThat(dto.isShared()).isTrue();
            assertThat(dto.getUpdaterId()).isEqualTo("admin");
            assertThat(dto.isCopied()).isTrue();
            assertThat(dto.getSourceAddressId()).isEqualTo(5L);
            assertThat(dto.getPartnershipOpportunityId()).isEqualTo(100L);
        }

        @Test
        @DisplayName("all setters should work")
        void allSettersShouldWork() {
            AddressDtoOut dto = new AddressDtoOut();
            LocalDateTime now = LocalDateTime.now();

            dto.setId(1L);
            dto.setUserId(10L);
            dto.setStreet("Test Street");
            dto.setCity("Krakow");
            dto.setPostalCode("30-001");
            dto.setCountry("Poland");
            dto.setState("Malopolskie");
            dto.setAdditionalInfo("Floor 2");
            dto.setAddressType("BUSINESS");
            dto.setPrimary(true);
            dto.setCreatedTime(now);
            dto.setLastUpdateTime(now);
            dto.setSourceType(AddressSourceType.CUSTOM);
            dto.setShared(true);
            dto.setUpdaterId("user-123");
            dto.setCopied(true);
            dto.setSourceAddressId(99L);
            dto.setPartnershipOpportunityId(50L);

            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getUserId()).isEqualTo(10L);
            assertThat(dto.getStreet()).isEqualTo("Test Street");
            assertThat(dto.getCity()).isEqualTo("Krakow");
            assertThat(dto.getPostalCode()).isEqualTo("30-001");
            assertThat(dto.getCountry()).isEqualTo("Poland");
            assertThat(dto.getState()).isEqualTo("Malopolskie");
            assertThat(dto.getAdditionalInfo()).isEqualTo("Floor 2");
            assertThat(dto.getAddressType()).isEqualTo("BUSINESS");
            assertThat(dto.isPrimary()).isTrue();
            assertThat(dto.getSourceType()).isEqualTo(AddressSourceType.CUSTOM);
            assertThat(dto.isShared()).isTrue();
            assertThat(dto.getUpdaterId()).isEqualTo("user-123");
            assertThat(dto.isCopied()).isTrue();
            assertThat(dto.getSourceAddressId()).isEqualTo(99L);
            assertThat(dto.getPartnershipOpportunityId()).isEqualTo(50L);
        }

        @ParameterizedTest
        @EnumSource(AddressSourceType.class)
        @DisplayName("should accept all source types")
        void shouldAcceptAllSourceTypes(AddressSourceType sourceType) {
            AddressDtoOut dto = new AddressDtoOut();
            dto.setSourceType(sourceType);

            assertThat(dto.getSourceType()).isEqualTo(sourceType);
        }
    }

    @Nested
    @DisplayName("CurrencyDto Tests")
    class CurrencyDtoTests {

        @Test
        @DisplayName("all fields should have getters and setters")
        void allFieldsShouldHaveGettersAndSetters() {
            CurrencyDto dto = new CurrencyDto();

            dto.setId(1L);
            dto.setName("Polish Zloty");
            dto.setIsoCode("PLN");
            dto.setSign("zl");
            dto.setCountryCode("PL");

            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getName()).isEqualTo("Polish Zloty");
            assertThat(dto.getIsoCode()).isEqualTo("PLN");
            assertThat(dto.getSign()).isEqualTo("zl");
            assertThat(dto.getCountryCode()).isEqualTo("PL");
        }

        @Test
        @DisplayName("should accept null values")
        void shouldAcceptNullValues() {
            CurrencyDto dto = new CurrencyDto();

            assertThat(dto.getId()).isNull();
            assertThat(dto.getName()).isNull();
            assertThat(dto.getIsoCode()).isNull();
            assertThat(dto.getSign()).isNull();
            assertThat(dto.getCountryCode()).isNull();
        }
    }

    @Nested
    @DisplayName("CurrencyDtoOut Tests")
    class CurrencyDtoOutTests {

        @Test
        @DisplayName("builder should create valid object")
        void builderShouldCreateValidObject() {
            CurrencyDtoOut dto = CurrencyDtoOut.builder()
                    .id(1L)
                    .name("Polski Zloty")
                    .originalName("Polish Zloty")
                    .isoCode("PLN")
                    .sign("zl")
                    .countryCode("PL")
                    .build();

            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getName()).isEqualTo("Polski Zloty");
            assertThat(dto.getOriginalName()).isEqualTo("Polish Zloty");
            assertThat(dto.getIsoCode()).isEqualTo("PLN");
            assertThat(dto.getSign()).isEqualTo("zl");
            assertThat(dto.getCountryCode()).isEqualTo("PL");
        }

        @Test
        @DisplayName("no-args constructor should work")
        void noArgsConstructorShouldWork() {
            CurrencyDtoOut dto = new CurrencyDtoOut();
            assertThat(dto).isNotNull();
        }

        @Test
        @DisplayName("all-args constructor should work")
        void allArgsConstructorShouldWork() {
            CurrencyDtoOut dto = new CurrencyDtoOut(1L, "Euro", "Euro", "EUR", "E", "EU");

            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getName()).isEqualTo("Euro");
            assertThat(dto.getIsoCode()).isEqualTo("EUR");
        }

        @Test
        @DisplayName("equals and hashCode should work")
        void equalsAndHashCodeShouldWork() {
            CurrencyDtoOut dto1 = CurrencyDtoOut.builder()
                    .id(1L)
                    .isoCode("PLN")
                    .build();

            CurrencyDtoOut dto2 = CurrencyDtoOut.builder()
                    .id(1L)
                    .isoCode("PLN")
                    .build();

            assertThat(dto1).isEqualTo(dto2);
            assertThat(dto1.hashCode()).isEqualTo(dto2.hashCode());
        }

        @Test
        @DisplayName("toString should include fields")
        void toStringShouldIncludeFields() {
            CurrencyDtoOut dto = CurrencyDtoOut.builder()
                    .id(1L)
                    .isoCode("PLN")
                    .build();

            String str = dto.toString();
            assertThat(str).contains("PLN");
        }
    }

    @Nested
    @DisplayName("CoopDto Tests")
    class CoopDtoTests {

        @Test
        @DisplayName("no-args constructor should work")
        void noArgsConstructorShouldWork() {
            CoopDto dto = new CoopDto();
            assertThat(dto).isNotNull();
        }

        @Test
        @DisplayName("all-args constructor should work")
        void allArgsConstructorShouldWork() {
            CoopDto dto = new CoopDto(
                    1L, "John", "Doe", "Company ABC", "john@example.com",
                    "instagram_user", 10000, "https://avatar.com/john.jpg",
                    "https://avatar.com/company.jpg", 100L, OpportunityStatus.ACCEPTED_BY_INFLUENCER,
                    "Campaign Title", "Note text", RateStatus.POSITIVE,
                    RateStatus.DEFAULT, 5L, 1L
            );

            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getInfluencerFirstName()).isEqualTo("John");
            assertThat(dto.getInfluencerLastName()).isEqualTo("Doe");
            assertThat(dto.getCompanyName()).isEqualTo("Company ABC");
            assertThat(dto.getInfluencerEmail()).isEqualTo("john@example.com");
            assertThat(dto.getInfluencerInstagramId()).isEqualTo("instagram_user");
            assertThat(dto.getFollowersAmount()).isEqualTo(10000);
            assertThat(dto.getInfluencerAvatarUrl()).isEqualTo("https://avatar.com/john.jpg");
            assertThat(dto.getCompanyAvatarUrl()).isEqualTo("https://avatar.com/company.jpg");
            assertThat(dto.getAppliedOpportunityId()).isEqualTo(100L);
            assertThat(dto.getAppliedOpportunityStatus()).isEqualTo(OpportunityStatus.ACCEPTED_BY_INFLUENCER);
            assertThat(dto.getPartnershipOpportunityTitle()).isEqualTo("Campaign Title");
            assertThat(dto.getAppliedOpportunityNote()).isEqualTo("Note text");
            assertThat(dto.getInfluencerRateStatus()).isEqualTo(RateStatus.POSITIVE);
            assertThat(dto.getCompanyRateStatus()).isEqualTo(RateStatus.DEFAULT);
            assertThat(dto.getPositiveRatesAmount()).isEqualTo(5L);
            assertThat(dto.getNegativeRatesAmount()).isEqualTo(1L);
        }

        @Test
        @DisplayName("all setters should work")
        void allSettersShouldWork() {
            CoopDto dto = new CoopDto();

            dto.setId(1L);
            dto.setInfluencerFirstName("Jane");
            dto.setInfluencerLastName("Smith");
            dto.setCompanyName("Test Company");
            dto.setInfluencerEmail("jane@test.com");
            dto.setInfluencerInstagramId("jane_insta");
            dto.setFollowersAmount(5000);
            dto.setInfluencerAvatarUrl("https://img.com/jane.jpg");
            dto.setCompanyAvatarUrl("https://img.com/company.jpg");
            dto.setAppliedOpportunityId(50L);
            dto.setAppliedOpportunityStatus(OpportunityStatus.DONE);
            dto.setPartnershipOpportunityTitle("New Campaign");
            dto.setAppliedOpportunityNote("Some note");
            dto.setInfluencerRateStatus(RateStatus.DEFAULT);
            dto.setCompanyRateStatus(RateStatus.NEGATIVE);
            dto.setPositiveRatesAmount(10L);
            dto.setNegativeRatesAmount(2L);

            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getInfluencerFirstName()).isEqualTo("Jane");
            assertThat(dto.getInfluencerLastName()).isEqualTo("Smith");
            assertThat(dto.getCompanyName()).isEqualTo("Test Company");
            assertThat(dto.getInfluencerEmail()).isEqualTo("jane@test.com");
            assertThat(dto.getInfluencerInstagramId()).isEqualTo("jane_insta");
            assertThat(dto.getFollowersAmount()).isEqualTo(5000);
            assertThat(dto.getInfluencerAvatarUrl()).isEqualTo("https://img.com/jane.jpg");
            assertThat(dto.getCompanyAvatarUrl()).isEqualTo("https://img.com/company.jpg");
            assertThat(dto.getAppliedOpportunityId()).isEqualTo(50L);
            assertThat(dto.getAppliedOpportunityStatus()).isEqualTo(OpportunityStatus.DONE);
            assertThat(dto.getPartnershipOpportunityTitle()).isEqualTo("New Campaign");
            assertThat(dto.getAppliedOpportunityNote()).isEqualTo("Some note");
            assertThat(dto.getInfluencerRateStatus()).isEqualTo(RateStatus.DEFAULT);
            assertThat(dto.getCompanyRateStatus()).isEqualTo(RateStatus.NEGATIVE);
            assertThat(dto.getPositiveRatesAmount()).isEqualTo(10L);
            assertThat(dto.getNegativeRatesAmount()).isEqualTo(2L);
        }

        @ParameterizedTest
        @EnumSource(OpportunityStatus.class)
        @DisplayName("should accept all opportunity statuses")
        void shouldAcceptAllOpportunityStatuses(OpportunityStatus status) {
            CoopDto dto = new CoopDto();
            dto.setAppliedOpportunityStatus(status);

            assertThat(dto.getAppliedOpportunityStatus()).isEqualTo(status);
        }

        @ParameterizedTest
        @EnumSource(RateStatus.class)
        @DisplayName("should accept all rate statuses")
        void shouldAcceptAllRateStatuses(RateStatus status) {
            CoopDto dto = new CoopDto();
            dto.setInfluencerRateStatus(status);
            dto.setCompanyRateStatus(status);

            assertThat(dto.getInfluencerRateStatus()).isEqualTo(status);
            assertThat(dto.getCompanyRateStatus()).isEqualTo(status);
        }
    }

    @Nested
    @DisplayName("CoopFilter Tests")
    class CoopFilterTests {

        @Test
        @DisplayName("all fields should have getters and setters")
        void allFieldsShouldHaveGettersAndSetters() {
            CoopFilter filter = new CoopFilter();

            filter.setOpportunityStatuses(List.of(OpportunityStatus.ACCEPTED_BY_INFLUENCER, OpportunityStatus.DONE));
            filter.setCompanyId(1L);
            filter.setInfluencerId(2L);
            filter.setFilterRateStatus(RateStatus.POSITIVE);
            filter.setMinFollowers(1000);
            filter.setMaxFollowers(100000);
            filter.setMinPositiveRates(5L);
            filter.setPartnershipOpportunityId(10L);

            assertThat(filter.getOpportunityStatuses()).containsExactly(OpportunityStatus.ACCEPTED_BY_INFLUENCER, OpportunityStatus.DONE);
            assertThat(filter.getCompanyId()).isEqualTo(1L);
            assertThat(filter.getInfluencerId()).isEqualTo(2L);
            assertThat(filter.getFilterRateStatus()).isEqualTo(RateStatus.POSITIVE);
            assertThat(filter.getMinFollowers()).isEqualTo(1000);
            assertThat(filter.getMaxFollowers()).isEqualTo(100000);
            assertThat(filter.getMinPositiveRates()).isEqualTo(5L);
            assertThat(filter.getPartnershipOpportunityId()).isEqualTo(10L);
        }

        @Test
        @DisplayName("should accept null values")
        void shouldAcceptNullValues() {
            CoopFilter filter = new CoopFilter();

            assertThat(filter.getOpportunityStatuses()).isNull();
            assertThat(filter.getCompanyId()).isNull();
            assertThat(filter.getInfluencerId()).isNull();
            assertThat(filter.getFilterRateStatus()).isNull();
            assertThat(filter.getMinFollowers()).isNull();
            assertThat(filter.getMaxFollowers()).isNull();
        }

        @Test
        @DisplayName("should accept empty list for statuses")
        void shouldAcceptEmptyListForStatuses() {
            CoopFilter filter = new CoopFilter();
            filter.setOpportunityStatuses(List.of());

            assertThat(filter.getOpportunityStatuses()).isEmpty();
        }
    }

    @Nested
    @DisplayName("Views Interface Tests")
    class ViewsInterfaceTests {

        @Test
        @DisplayName("Basic interface should exist")
        void basicInterfaceShouldExist() {
            assertThat(Views.Basic.class).isInterface();
        }

        @Test
        @DisplayName("Ratings should extend Basic")
        void ratingsShouldExtendBasic() {
            assertThat(Views.Basic.class.isAssignableFrom(Views.Ratings.class)).isTrue();
        }

        @Test
        @DisplayName("Registration should extend Ratings")
        void registrationShouldExtendRatings() {
            assertThat(Views.Ratings.class.isAssignableFrom(Views.Registration.class)).isTrue();
        }

        @Test
        @DisplayName("InProgress_InfluencerView should extend Basic")
        void inProgressInfluencerViewShouldExtendBasic() {
            assertThat(Views.Basic.class.isAssignableFrom(Views.InProgress_InfluencerView.class)).isTrue();
        }

        @Test
        @DisplayName("InProgress_CompanyView should extend Basic")
        void inProgressCompanyViewShouldExtendBasic() {
            assertThat(Views.Basic.class.isAssignableFrom(Views.InProgress_CompanyView.class)).isTrue();
        }

        @Test
        @DisplayName("InProgress_AdminView should extend both InfluencerView and CompanyView")
        void inProgressAdminViewShouldExtendBoth() {
            assertThat(Views.InProgress_InfluencerView.class.isAssignableFrom(Views.InProgress_AdminView.class)).isTrue();
            assertThat(Views.InProgress_CompanyView.class.isAssignableFrom(Views.InProgress_AdminView.class)).isTrue();
        }
    }

    @Nested
    @DisplayName("PublicProfileDto Sealed Interface Tests")
    class PublicProfileDtoTests {

        @Test
        @DisplayName("PublicProfileDto should be sealed interface")
        void publicProfileDtoShouldBeSealedInterface() {
            assertThat(PublicProfileDto.class).isInterface();
            assertThat(PublicProfileDto.class.isSealed()).isTrue();
        }

        @Test
        @DisplayName("InfluencerPublicProfileDto should implement PublicProfileDto")
        void influencerPublicProfileDtoShouldImplementPublicProfileDto() {
            assertThat(PublicProfileDto.class.isAssignableFrom(InfluencerPublicProfileDto.class)).isTrue();
        }

        @Test
        @DisplayName("CompanyPublicProfileDto should implement PublicProfileDto")
        void companyPublicProfileDtoShouldImplementPublicProfileDto() {
            assertThat(PublicProfileDto.class.isAssignableFrom(CompanyPublicProfileDto.class)).isTrue();
        }
    }

    @Nested
    @DisplayName("InfluencerPublicProfileDto Tests")
    class InfluencerPublicProfileDtoTests {

        @Test
        @DisplayName("all fields should have getters and setters")
        void allFieldsShouldHaveGettersAndSetters() {
            InfluencerPublicProfileDto dto = new InfluencerPublicProfileDto();
            LocalDateTime now = LocalDateTime.now();

            dto.setId(1L);
            dto.setName("Influencer Name");
            dto.setProfilePicture("https://img.com/profile.jpg");
            dto.setCreatedTime(now);
            dto.setPlatformName("Instagram");
            dto.setDisplayName("@influencer");
            dto.setProfileUrl("https://instagram.com/influencer");
            dto.setAccountStatus(new AccountStatusDtoOut());
            dto.setFollowersCount(10000);
            dto.setPremium(true);

            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getName()).isEqualTo("Influencer Name");
            assertThat(dto.getProfilePicture()).isEqualTo("https://img.com/profile.jpg");
            assertThat(dto.getCreatedTime()).isEqualTo(now);
            assertThat(dto.getPlatformName()).isEqualTo("Instagram");
            assertThat(dto.getDisplayName()).isEqualTo("@influencer");
            assertThat(dto.getProfileUrl()).isEqualTo("https://instagram.com/influencer");
            assertThat(dto.getAccountStatus()).isNotNull();
            assertThat(dto.getFollowersCount()).isEqualTo(10000);
            assertThat(dto.getPremium()).isTrue();
        }

        @Test
        @DisplayName("should accept null values")
        void shouldAcceptNullValues() {
            InfluencerPublicProfileDto dto = new InfluencerPublicProfileDto();

            assertThat(dto.getId()).isNull();
            assertThat(dto.getName()).isNull();
            assertThat(dto.getPremium()).isNull();
            assertThat(dto.getFollowersCount()).isNull();
        }
    }

    @Nested
    @DisplayName("CompanyPublicProfileDto Tests")
    class CompanyPublicProfileDtoTests {

        @Test
        @DisplayName("all fields should have getters and setters")
        void allFieldsShouldHaveGettersAndSetters() {
            CompanyPublicProfileDto dto = new CompanyPublicProfileDto();

            dto.setId(1L);
            dto.setName("Company Name");
            dto.setProfilePicture("https://img.com/company.jpg");
            dto.setAddresses(List.of());
            dto.setDescription("Company description");
            dto.setAccountStatus(new AccountStatusDtoOut());
            dto.setWebsite("https://company.com");
            dto.setPremium(true);

            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getName()).isEqualTo("Company Name");
            assertThat(dto.getProfilePicture()).isEqualTo("https://img.com/company.jpg");
            assertThat(dto.getAddresses()).isEmpty();
            assertThat(dto.getDescription()).isEqualTo("Company description");
            assertThat(dto.getAccountStatus()).isNotNull();
            assertThat(dto.getWebsite()).isEqualTo("https://company.com");
            assertThat(dto.getPremium()).isTrue();
        }

        @Test
        @DisplayName("should accept null values")
        void shouldAcceptNullValues() {
            CompanyPublicProfileDto dto = new CompanyPublicProfileDto();

            assertThat(dto.getId()).isNull();
            assertThat(dto.getName()).isNull();
            assertThat(dto.getAddresses()).isNull();
            assertThat(dto.getPremium()).isNull();
        }
    }

    @Nested
    @DisplayName("InfluencerForCompanyProfileDto Tests")
    class InfluencerForCompanyProfileDtoTests {

        @Test
        @DisplayName("should extend InfluencerPublicProfileDto")
        void shouldExtendInfluencerPublicProfileDto() {
            assertThat(InfluencerPublicProfileDto.class.isAssignableFrom(InfluencerForCompanyProfileDto.class)).isTrue();
        }

        @Test
        @DisplayName("all fields should have getters and setters")
        void allFieldsShouldHaveGettersAndSetters() {
            InfluencerForCompanyProfileDto dto = new InfluencerForCompanyProfileDto();

            dto.setId(1L);
            dto.setName("Influencer");
            dto.setEmail("influencer@example.com");
            dto.setFirstName("John");
            dto.setLastName("Doe");
            dto.setAddresses(List.of());
            dto.setPhoneNumber("+48123456789");
            dto.setSocialConnections(List.of());
            dto.setPremium(false);

            assertThat(dto.getId()).isEqualTo(1L);
            assertThat(dto.getName()).isEqualTo("Influencer");
            assertThat(dto.getEmail()).isEqualTo("influencer@example.com");
            assertThat(dto.getFirstName()).isEqualTo("John");
            assertThat(dto.getLastName()).isEqualTo("Doe");
            assertThat(dto.getAddresses()).isEmpty();
            assertThat(dto.getPhoneNumber()).isEqualTo("+48123456789");
            assertThat(dto.getSocialConnections()).isEmpty();
            assertThat(dto.getPremium()).isFalse();
        }

        @Test
        @DisplayName("inherited fields from parent should work")
        void inheritedFieldsFromParentShouldWork() {
            InfluencerForCompanyProfileDto dto = new InfluencerForCompanyProfileDto();
            LocalDateTime now = LocalDateTime.now();

            dto.setProfilePicture("https://img.com/profile.jpg");
            dto.setCreatedTime(now);
            dto.setPlatformName("TikTok");
            dto.setDisplayName("@tiktok_user");
            dto.setProfileUrl("https://tiktok.com/@user");
            dto.setFollowersCount(50000);

            assertThat(dto.getProfilePicture()).isEqualTo("https://img.com/profile.jpg");
            assertThat(dto.getCreatedTime()).isEqualTo(now);
            assertThat(dto.getPlatformName()).isEqualTo("TikTok");
            assertThat(dto.getDisplayName()).isEqualTo("@tiktok_user");
            assertThat(dto.getProfileUrl()).isEqualTo("https://tiktok.com/@user");
            assertThat(dto.getFollowersCount()).isEqualTo(50000);
        }
    }

    @Nested
    @DisplayName("AccountStatusDtoOut Tests")
    class AccountStatusDtoOutTests {

        @Test
        @DisplayName("should be instantiable")
        void shouldBeInstantiable() {
            AccountStatusDtoOut dto = new AccountStatusDtoOut();
            assertThat(dto).isNotNull();
        }
    }
}
