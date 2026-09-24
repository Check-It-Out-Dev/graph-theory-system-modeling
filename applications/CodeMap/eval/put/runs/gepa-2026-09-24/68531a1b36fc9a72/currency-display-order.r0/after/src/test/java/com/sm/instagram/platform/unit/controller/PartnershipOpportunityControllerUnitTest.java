package com.sm.instagram.platform.unit.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sm.instagram.platform.common.exceptions.InsufficientPermissionsException;
import com.sm.instagram.platform.common.exceptions.ResourceNotFoundException;
import com.sm.instagram.platform.common.exceptions.handlers.AuthenticationExceptionHandler;
import com.sm.instagram.platform.common.exceptions.handlers.BusinessExceptionHandler;
import com.sm.instagram.platform.common.translation.TranslationService;
import com.sm.instagram.platform.currency.Currency;
import com.sm.instagram.platform.partnershipopportunities.*;
import com.sm.instagram.platform.user.CompanyPublicProfileDto;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.context.annotation.Import;
import org.springframework.test.context.ContextConfiguration;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.PageRequest;
import org.springframework.http.MediaType;
import org.springframework.security.test.context.support.WithMockUser;
import org.springframework.test.web.servlet.MockMvc;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import static org.hamcrest.Matchers.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.*;
import static org.mockito.Mockito.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

/**
 * Unit tests for PartnershipOpportunityController using @WebMvcTest.
 * Tests HTTP endpoints, request validation, and response formatting.
 * Does NOT load full Spring context - only web layer with mocked service.
 */
@WebMvcTest(controllers = PartnershipOpportunityController.class)
@AutoConfigureMockMvc(addFilters = false)
@ContextConfiguration(classes = {
        PartnershipOpportunityController.class,
        TestControllerSecurityConfig.class,
        BusinessExceptionHandler.class,
        AuthenticationExceptionHandler.class
})
@DisplayName("PartnershipOpportunityController")
class PartnershipOpportunityControllerUnitTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @MockBean
    private PartnershipOpportunityService opportunitiesService;

    @MockBean
    private TranslationService translationService;

    @MockBean
    private org.springframework.context.MessageSource messageSource;

    // Test fixtures
    private PartnershipOpportunityDtoOut testOpportunityDto;
    private PartnershipOpportunityDtoIn testOpportunityDtoIn;

    @BeforeEach
    void setUp() {
        // Create test output DTO
        testOpportunityDto = new PartnershipOpportunityDtoOut();
        testOpportunityDto.setId(100L);
        testOpportunityDto.setName("Test Campaign");
        testOpportunityDto.setTitle("Campaign Title");
        testOpportunityDto.setDetails("Campaign details here");
        testOpportunityDto.setFollowersMin(1000L);
        testOpportunityDto.setFollowersMax(50000L);
        testOpportunityDto.setCompensationAmountMin(100);
        testOpportunityDto.setCompensationAmountMax(500);
        testOpportunityDto.setStartDate(LocalDateTime.now());
        testOpportunityDto.setEndDate(LocalDateTime.now().plusMonths(1));
        testOpportunityDto.setActive(true);
        testOpportunityDto.setCreatedTime(LocalDateTime.now());
        testOpportunityDto.setLastUpdateTime(LocalDateTime.now());

        // Set company
        CompanyPublicProfileDto companyDto = new CompanyPublicProfileDto();
        companyDto.setId(1L);
        companyDto.setName("Test Company");
        testOpportunityDto.setCompany(companyDto);

        // Create test input DTO - all required fields must be set for validation
        testOpportunityDtoIn = new PartnershipOpportunityDtoIn();
        testOpportunityDtoIn.setName("New Campaign");
        testOpportunityDtoIn.setCity("Warsaw");  // Required field
        testOpportunityDtoIn.setTitle("New Campaign Title");
        testOpportunityDtoIn.setDetails("New campaign details");
        testOpportunityDtoIn.setFollowersMin(1000L);
        testOpportunityDtoIn.setFollowersMax(50000L);
        testOpportunityDtoIn.setCompensationAmountMin(100);
        testOpportunityDtoIn.setCompensationAmountMax(500);
        testOpportunityDtoIn.setStartDate(LocalDateTime.now());
        testOpportunityDtoIn.setEndDate(LocalDateTime.now().plusMonths(1));
        testOpportunityDtoIn.setActive(true);
        testOpportunityDtoIn.setCompany(1L);
    }

    @Nested
    @DisplayName("GET /partnership-opportunity/{id}")
    class GetById {

        @Test
        @WithMockUser(authorities = "COMPANY")
        @DisplayName("should return opportunity when authenticated and found")
        void shouldReturnOpportunityWhenAuthenticatedAndFound() throws Exception {
            // Given
            when(opportunitiesService.findByIdAsDto(eq(100L), any(Locale.class)))
                    .thenReturn(testOpportunityDto);
            when(opportunitiesService.findById(100L))
                    .thenReturn(new PartnershipOpportunity());

            // When/Then
            mockMvc.perform(get("/partnership-opportunity/100")
                            .accept(MediaType.APPLICATION_JSON)
                            .header("Accept-Language", "en"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.id").value(100))
                    .andExpect(jsonPath("$.name").value("Test Campaign"))
                    .andExpect(jsonPath("$.title").value("Campaign Title"))
                    .andExpect(jsonPath("$.active").value(true));

            verify(opportunitiesService).findByIdAsDto(eq(100L), any(Locale.class));
        }

        @Test
        @WithMockUser(authorities = "INFLUENCER")
        @DisplayName("should return opportunity for influencer")
        void shouldReturnOpportunityForInfluencer() throws Exception {
            // Given
            when(opportunitiesService.findByIdAsDto(eq(100L), any(Locale.class)))
                    .thenReturn(testOpportunityDto);
            when(opportunitiesService.findById(100L))
                    .thenReturn(new PartnershipOpportunity());

            // When/Then
            mockMvc.perform(get("/partnership-opportunity/100")
                            .accept(MediaType.APPLICATION_JSON))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.id").value(100));
        }

        // Note: Authentication tests (401) are handled in integration tests with full security context

        @Test
        @WithMockUser(authorities = "COMPANY")
        @DisplayName("should return 404 when opportunity not found")
        void shouldReturn404WhenNotFound() throws Exception {
            // Given
            when(opportunitiesService.findByIdAsDto(eq(999L), any(Locale.class)))
                    .thenThrow(new ResourceNotFoundException("error.business.item_not_found", "Opportunity"));

            // When/Then
            mockMvc.perform(get("/partnership-opportunity/999")
                            .accept(MediaType.APPLICATION_JSON))
                    .andExpect(status().isNotFound());
        }

        @Test
        @WithMockUser(authorities = "COMPANY")
        @DisplayName("should return 403 when user cannot view opportunity")
        void shouldReturn403WhenCannotView() throws Exception {
            // Given
            when(opportunitiesService.findByIdAsDto(eq(100L), any(Locale.class)))
                    .thenThrow(new InsufficientPermissionsException(
                            "error.auth.insufficient_permissions",
                            "user-uid",
                            "findById",
                            "PartnershipOpportunity#100"));

            // When/Then
            mockMvc.perform(get("/partnership-opportunity/100")
                            .accept(MediaType.APPLICATION_JSON))
                    .andExpect(status().isForbidden());
        }
    }

    @Nested
    @DisplayName("GET /partnership-opportunity/paged")
    class GetPaged {

        @Test
        @WithMockUser(authorities = "INFLUENCER")
        @DisplayName("should return paginated opportunities")
        void shouldReturnPaginatedOpportunities() throws Exception {
            // Given
            Page<PartnershipOpportunityDtoOut> page = new PageImpl<>(
                    List.of(testOpportunityDto),
                    PageRequest.of(0, 10),
                    1
            );

            when(opportunitiesService.getDataPagedAndFilteredAsDtos(
                    any(), anyMap(), any(Locale.class)))
                    .thenReturn(page);
            // Create proper entity with ID to avoid NPE
            PartnershipOpportunity entity = new PartnershipOpportunity();
            entity.setId(100L);
            when(opportunitiesService.getDataPagedAndFiltered(any(), anyMap()))
                    .thenReturn(new PageImpl<>(List.of(entity)));

            // When/Then
            mockMvc.perform(get("/partnership-opportunity/paged")
                            .param("page", "0")
                            .param("size", "10")
                            .accept(MediaType.APPLICATION_JSON))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.content").isArray())
                    .andExpect(jsonPath("$.content", hasSize(1)))
                    .andExpect(jsonPath("$.content[0].id").value(100))
                    .andExpect(jsonPath("$.totalElements").value(1))
                    .andExpect(jsonPath("$.totalPages").value(1));
        }

        @Test
        @WithMockUser(authorities = "COMPANY")
        @DisplayName("should return empty page when no opportunities")
        void shouldReturnEmptyPageWhenNoOpportunities() throws Exception {
            // Given
            Page<PartnershipOpportunityDtoOut> emptyPage = new PageImpl<>(
                    List.of(),
                    PageRequest.of(0, 10),
                    0
            );

            when(opportunitiesService.getDataPagedAndFilteredAsDtos(
                    any(), anyMap(), any(Locale.class)))
                    .thenReturn(emptyPage);
            when(opportunitiesService.getDataPagedAndFiltered(any(), anyMap()))
                    .thenReturn(new PageImpl<>(List.of()));

            // When/Then
            mockMvc.perform(get("/partnership-opportunity/paged")
                            .param("page", "0")
                            .param("size", "10")
                            .accept(MediaType.APPLICATION_JSON))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.content").isArray())
                    .andExpect(jsonPath("$.content", hasSize(0)))
                    .andExpect(jsonPath("$.totalElements").value(0));
        }

        @Test
        @WithMockUser(authorities = "INFLUENCER")
        @DisplayName("should include currency on list rows (FE contract gap)")
        void shouldIncludeCurrencyOnListRows() throws Exception {
            // Given — list DTO without currency, entity carrying one: the
            // controller loop must patch it like it patches compensationType.
            Page<PartnershipOpportunityDtoOut> page = new PageImpl<>(
                    List.of(testOpportunityDto),
                    PageRequest.of(0, 10),
                    1
            );
            when(opportunitiesService.getDataPagedAndFilteredAsDtos(
                    any(), anyMap(), any(Locale.class)))
                    .thenReturn(page);

            Currency pln = new Currency();
            pln.setId(1L);
            pln.setName("Polish zloty");
            pln.setIsoCode("PLN");
            pln.setSign("zl");
            pln.setDisplayOrder(1);
            PartnershipOpportunity entity = new PartnershipOpportunity();
            entity.setId(100L);
            entity.setCurrency(pln);
            when(opportunitiesService.getDataPagedAndFiltered(any(), anyMap()))
                    .thenReturn(new PageImpl<>(List.of(entity)));
            when(translationService.translateCurrency(eq("PLN"), any(Locale.class)))
                    .thenReturn("zloty polski");

            // When/Then
            mockMvc.perform(get("/partnership-opportunity/paged")
                            .param("page", "0")
                            .param("size", "10")
                            .accept(MediaType.APPLICATION_JSON))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.content[0].currency.isoCode").value("PLN"))
                    .andExpect(jsonPath("$.content[0].currency.name").value("zloty polski"))
                    .andExpect(jsonPath("$.content[0].currency.sign").value("zl"))
                    .andExpect(jsonPath("$.content[0].currency.displayOrder").value(1));
        }

        // Note: Authentication tests (401) are handled in integration tests
    }

    @Nested
    @DisplayName("POST /partnership-opportunity")
    class CreateOpportunity {

        @Test
        @WithMockUser(authorities = "COMPANY")
        @DisplayName("should create opportunity when valid data")
        void shouldCreateOpportunityWhenValidData() throws Exception {
            // Given
            when(opportunitiesService.saveFromDtoAsDto(any(PartnershipOpportunityDtoIn.class), any(Locale.class)))
                    .thenReturn(testOpportunityDto);

            // When/Then
            mockMvc.perform(post("/partnership-opportunity")
                                                        .contentType(MediaType.APPLICATION_JSON)
                            .content(objectMapper.writeValueAsString(testOpportunityDtoIn)))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.id").value(100))
                    .andExpect(jsonPath("$.name").value("Test Campaign"));

            verify(opportunitiesService).saveFromDtoAsDto(
                    any(PartnershipOpportunityDtoIn.class), any(Locale.class));
        }

        @Test
        @WithMockUser(authorities = "INFLUENCER")
        @DisplayName("should return 403 when influencer tries to create")
        void shouldReturn403WhenInfluencerTriesToCreate() throws Exception {
            // Given - Influencer should not be able to create opportunities
            when(opportunitiesService.saveFromDtoAsDto(any(PartnershipOpportunityDtoIn.class), any(Locale.class)))
                    .thenThrow(new InsufficientPermissionsException(
                            "error.auth.insufficient_permissions",
                            "influencer-uid",
                            "saveFromDto",
                            "PartnershipOpportunity"));

            // When/Then
            mockMvc.perform(post("/partnership-opportunity")
                                                        .contentType(MediaType.APPLICATION_JSON)
                            .content(objectMapper.writeValueAsString(testOpportunityDtoIn)))
                    .andExpect(status().isForbidden());
        }

        // Note: Authentication tests (401) are handled in integration tests
    }

    @Nested
    @DisplayName("PUT /partnership-opportunity/{id}")
    class UpdateOpportunity {

        @Test
        @WithMockUser(authorities = "COMPANY")
        @DisplayName("should update opportunity when authorized")
        void shouldUpdateOpportunityWhenAuthorized() throws Exception {
            // Given
            when(opportunitiesService.updateAsDto(eq(100L), any(PartnershipOpportunityDtoIn.class), any(Locale.class)))
                    .thenReturn(testOpportunityDto);

            // When/Then
            mockMvc.perform(put("/partnership-opportunity/100")
                                                        .contentType(MediaType.APPLICATION_JSON)
                            .content(objectMapper.writeValueAsString(testOpportunityDtoIn)))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.id").value(100));

            verify(opportunitiesService).updateAsDto(
                    eq(100L), any(PartnershipOpportunityDtoIn.class), any(Locale.class));
        }

        @Test
        @WithMockUser(authorities = "COMPANY")
        @DisplayName("should return 403 when not owner")
        void shouldReturn403WhenNotOwner() throws Exception {
            // Given
            when(opportunitiesService.updateAsDto(eq(100L), any(PartnershipOpportunityDtoIn.class), any(Locale.class)))
                    .thenThrow(new InsufficientPermissionsException(
                            "error.auth.insufficient_permissions",
                            "different-company",
                            "update",
                            "PartnershipOpportunity#100"));

            // When/Then
            mockMvc.perform(put("/partnership-opportunity/100")
                                                        .contentType(MediaType.APPLICATION_JSON)
                            .content(objectMapper.writeValueAsString(testOpportunityDtoIn)))
                    .andExpect(status().isForbidden());
        }

        @Test
        @WithMockUser(authorities = "ADMIN")
        @DisplayName("should allow admin to update any opportunity")
        void shouldAllowAdminToUpdateAny() throws Exception {
            // Given
            when(opportunitiesService.updateAsDto(eq(100L), any(PartnershipOpportunityDtoIn.class), any(Locale.class)))
                    .thenReturn(testOpportunityDto);

            // When/Then
            mockMvc.perform(put("/partnership-opportunity/100")
                                                        .contentType(MediaType.APPLICATION_JSON)
                            .content(objectMapper.writeValueAsString(testOpportunityDtoIn)))
                    .andExpect(status().isOk());
        }
    }

    @Nested
    @DisplayName("PATCH /partnership-opportunity/{id}")
    class PatchOpportunity {

        @Test
        @WithMockUser(authorities = "COMPANY")
        @DisplayName("should patch opportunity with partial data")
        void shouldPatchOpportunityWithPartialData() throws Exception {
            // Given
            Map<String, Object> updates = Map.of(
                    "title", "Updated Title",
                    "active", false
            );

            when(opportunitiesService.patchAsDto(eq(100L), anyMap(), any(Locale.class)))
                    .thenReturn(testOpportunityDto);

            // When/Then
            mockMvc.perform(patch("/partnership-opportunity/100")
                                                        .contentType(MediaType.APPLICATION_JSON)
                            .content(objectMapper.writeValueAsString(updates)))
                    .andExpect(status().isOk());

            verify(opportunitiesService).patchAsDto(eq(100L), anyMap(), any(Locale.class));
        }
    }

    @Nested
    @DisplayName("GET /partnership-opportunity/compensation/type")
    class GetCompensationTypes {

        @Test
        @WithMockUser(authorities = "COMPANY")
        @DisplayName("should return all compensation types")
        void shouldReturnAllCompensationTypes() throws Exception {
            // When/Then
            mockMvc.perform(get("/partnership-opportunity/compensation/type")
                            .accept(MediaType.APPLICATION_JSON))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$").isArray())
                    .andExpect(jsonPath("$", hasItem("CASH")))
                    .andExpect(jsonPath("$", hasItem("BARTER")));
        }

        @Test
        @WithMockUser(authorities = "INFLUENCER")
        @DisplayName("should return compensation types for influencer")
        void shouldReturnCompensationTypesForInfluencer() throws Exception {
            mockMvc.perform(get("/partnership-opportunity/compensation/type")
                            .accept(MediaType.APPLICATION_JSON))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$").isArray());
        }

        // Note: Authentication tests (401) are handled in integration tests
    }

    @Nested
    @DisplayName("Accept-Language Header Handling")
    class LocaleHandling {

        @Test
        @WithMockUser(authorities = "COMPANY")
        @DisplayName("should use Accept-Language header for translations")
        void shouldUseAcceptLanguageHeader() throws Exception {
            // Given
            when(opportunitiesService.findByIdAsDto(eq(100L), any(Locale.class)))
                    .thenReturn(testOpportunityDto);
            when(opportunitiesService.findById(100L))
                    .thenReturn(new PartnershipOpportunity());

            // When/Then
            mockMvc.perform(get("/partnership-opportunity/100")
                            .accept(MediaType.APPLICATION_JSON)
                            .header("Accept-Language", "en-US"))
                    .andExpect(status().isOk());

            // Verify locale was passed correctly
            verify(opportunitiesService).findByIdAsDto(eq(100L), argThat(locale ->
                    locale.toLanguageTag().startsWith("en")));
        }

        @Test
        @WithMockUser(authorities = "COMPANY")
        @DisplayName("should default to Polish when no Accept-Language header")
        void shouldDefaultToPolishWhenNoHeader() throws Exception {
            // Given
            when(opportunitiesService.findByIdAsDto(eq(100L), any(Locale.class)))
                    .thenReturn(testOpportunityDto);
            when(opportunitiesService.findById(100L))
                    .thenReturn(new PartnershipOpportunity());

            // When/Then
            mockMvc.perform(get("/partnership-opportunity/100")
                            .accept(MediaType.APPLICATION_JSON))
                    .andExpect(status().isOk());

            // Verify Polish locale was used
            verify(opportunitiesService).findByIdAsDto(eq(100L), argThat(locale ->
                    locale.toLanguageTag().equals("pl")));
        }
    }
}
